# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:53:01 2024

@author: Scot Wheeler
"""

import pandas as pd
import json
import geopandas as gpd
import numpy as np
import geopy.distance
import atlite
import pypsa
import p_MinX_aux as aux
from tqdm.auto import tqdm
import warnings
import contextlib
import sys
import os
import math
import logging
import xarray as xr
import geodatasets as gds
from unidecode import unidecode

logging.basicConfig(level=logging.ERROR)

#%% micellaneous functions
# Define a context manager to suppress print statements and stderr for pypsa
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def clean_country_name(country_name):
    # Get country names without accents, spaces, apostrophes, or periods for loading files
    country_name_clean = unidecode(country_name)
    country_name_clean = country_name_clean.replace(" ", "")
    country_name_clean = country_name_clean.replace(".", "")
    country_name_clean = country_name_clean.replace("'", "")
    return country_name_clean

def combine_glaes_spider(country_names, spider_hex_path=None,
                         glaes_wind_path=None, glaes_pv_path=None):
    """
    Adapted from GeoH2-data-prep: https://github.com/ClimateCompatibleGrowth/GeoH2-data-prep

    Returns
    -------
    None.

    """
    # create a for loop that can loop through a list of country names
    for country_name in country_names:

        print(f"Combining GLAES and SPIDER data for {country_name}!")

        country_name_clean = clean_country_name(country_name)

        # Get paths
        if isinstance(spider_hex_path, type(None)):
            spider_hex_path = os.path.join("Data", "Spider", f"{country_name_clean}_hex.geojson")
        if isinstance(glaes_wind_path, type(None)):
            glaes_wind_path = os.path.join("Data", "GLAES", f"{country_name_clean}_turbine_placements.shp")
        if isinstance(glaes_pv_path, type(None)):
            glaes_pv_path = os.path.join("Data", "GLAES", f"{country_name_clean}_pv_placements.shp")
        save_path = os.path.join("Data", f"{country_name_clean}_hex_GeoXin.geojson")

        # Load all files and convert all to the country's CRS
        print(" - Loading files...")
        hexagons = gpd.read_file(spider_hex_path)
        wind_points = gpd.read_file(glaes_wind_path)
        pv_points = gpd.read_file(glaes_pv_path)
        hexagons.to_crs(pv_points.crs, inplace=True)

        print(" - Joining turbine locations...")
        # Spatial join the wind points to the polygons
        spatial_join = gpd.sjoin(wind_points, hexagons, how='left', predicate='within')

        # Group by polygon and count the points within each polygon
        wind_point_counts = spatial_join.groupby('index_right0').size()

        # Merge the point counts with the 'hex' GeoDataFrame based on the index
        hexagons['theo_turbines'] = wind_point_counts

        # If some polygons have no points, fill their 'point_count' with 0
        hexagons['theo_turbines'] = hexagons['theo_turbines'].fillna(0)

        print(" - Joining pv locations...")
        # Spatial join the pv points to the polygons
        spatial_join = gpd.sjoin(pv_points, hexagons, how='left', predicate='within')
        
        # Group by polygon and count the points within each polygon
        pv_point_counts = spatial_join.groupby('index_right0').size()

        # Merge the point counts with the 'hex' GeoDataFrame based on the index
        hexagons['theo_pv'] = pv_point_counts

        # If some polygons have no points, fill their 'point_count' with 0
        hexagons['theo_pv'] = hexagons['theo_pv'].fillna(0)

        print(" - Done! Saving to GeoJSON...")
        # Check if hex GeoDataFrame is empty before saving
        if not hexagons.empty:
            # Save the file
            hexagons.to_file(save_path, driver='GeoJSON', encoding='utf-8')
            print(" - Save complete!")
        else:
            print(" ! Hex GeoDataFrame is empty. This can happen when your country is much smaller than the hexagon size you have used in Spider. Please use smaller hexagons in Spider and retry. Not saving to GeoJSON.")

def determine_feedstock_sources(feedstock_points_gdf, hexagon_to_feedstock_distance_matrix, hix, feedstock_quantity):
    feedstock_ranked = feedstock_points_gdf.merge(hexagon_to_feedstock_distance_matrix.loc[hix,:], left_index=True, right_index=True).sort_values(by=hix)[["Annual capacity [kg/a]"]]
    feedstock_ranked["Cumulative [kg/a]"] = feedstock_ranked.cumsum()
    feedstock_ranked["Feedstock used [kg/a]"] = 0
    remaining_quantity = feedstock_quantity
    for f, feedstock in feedstock_ranked.iterrows():
        if remaining_quantity <= 0:
            break
        feedstock_used = min(feedstock_points_gdf.loc[f,"Annual capacity [kg/a]"], remaining_quantity)
        feedstock_ranked.loc[f,"Feedstock used [kg/a]"] = feedstock_used
        remaining_quantity -= feedstock_used
        
    
    feedstock = feedstock_ranked.loc[:feedstock_ranked[feedstock_ranked["Cumulative [kg/a]"] >= feedstock_quantity].index[0], :]
    feedstock_ranked_idxs = feedstock.index
    return feedstock, feedstock_ranked_idxs

#%% file imports

def import_parameter_files(tech_param_fpath="Parameters/technology_parameters.xlsx",
                           demand_param_fpath="Parameters/demand_parameters.xlsx",
                           country_param_fpath="Parameters/country_parameters.xlsx"):
    infra_data = pd.read_excel(tech_param_fpath,
                               sheet_name='Infra',
                               index_col='Infrastructure')

    global_data = pd.read_excel(tech_param_fpath,
                                sheet_name='Global',
                                index_col='Parameter'
                                ).squeeze("columns")

    mineral_data = pd.read_excel(tech_param_fpath,
                                sheet_name='Mining',
                                index_col='Parameter'
                                ).squeeze("columns")

    demand_center_list = pd.read_excel(demand_param_fpath,
                                       sheet_name='Demand centers',
                                       index_col='Demand center',
                                       )
    generation_center_list = pd.read_excel(demand_param_fpath,
                                       sheet_name='Generation centers',
                                       index_col='Generation center',
                                       )
    country_parameters = pd.read_excel(country_param_fpath,
                                        index_col='Country')
    
    return (infra_data, global_data, mineral_data, demand_center_list,
            generation_center_list, country_parameters)

def import_conversion_parameters(demand_states, conv_param_fpath="Parameters/conversion_parameters.xlsx"):
    conv_params=[]
    for demand_state in demand_states:
        conv_params.append(pd.read_excel(conv_param_fpath,
                                         sheet_name=demand_state,
                                         index_col='Parameter'))
    df = pd.concat(conv_params, axis=1)
    df.columns= demand_states
    return df

def get_pv_wind_profiles(hexagons_gdf,
                         weather_excel_path="Parameters/weather_parameters.xlsx"):
    weather_parameters = pd.read_excel(weather_excel_path,
                                       index_col = 'Parameters'
                                       ).squeeze('columns')
    weather_filename = weather_parameters['Filename']
    cutout = atlite.Cutout('Cutouts/' + weather_filename +'.nc')
    layout = cutout.uniform_layout()
    
    pv_profile = cutout.pv(
        panel= 'CSi',
        orientation='latitude_optimal',
        layout = layout,
        shapes = hexagons_gdf,
        per_unit = True
        )
    pv_profile = pv_profile.rename(dict(dim_0='hexagon'))

    wind_profile = cutout.wind(
        # Changed turbine type - was Vestas_V80_2MW_gridstreamer in first run
        # Other option being explored: NREL_ReferenceTurbine_2020ATB_4MW, Enercon_E126_7500kW
        turbine = 'NREL_ReferenceTurbine_2020ATB_4MW',
        layout = layout,
        shapes = hexagons_gdf,
        per_unit = True
        )
    wind_profile = wind_profile.rename(dict(dim_0='hexagon'))
    
    return pv_profile, wind_profile


def load_hexagons(country_name, country_parameters, how="centroid_within"):
    """
    Load hexagons produced by combine_glaes_spider function (or GeoH2_data_prep).
    You do not need to run assign_countries previously part of GeoH2_data_prep.

    Returns
    -------
    None.

    """
    country_name_clean = clean_country_name(country_name)
    
    inputpath = os.path.join("Data", f"{country_name_clean}_hex_GeoXin.geojson")
    outputpath = os.path.join("Data", f"{country_name_clean}_hex_GeoXin_with_country.geojson")
    
    # Load the Hexagon file in geopandas
    hexagons = gpd.read_file(inputpath)
    hexagons.to_crs(epsg=4326, inplace=True)
    # the use of centroid throws a UserWarning when a geographic CRS (e.g. EPSG=4326) is used.
    # ideally you should convert to a projected crs (ideally with equal area cylindrical?) then back again.
    warnings.filterwarnings('ignore', category=UserWarning)
    hexagon_centroids = gpd.GeoDataFrame(geometry=hexagons.centroid, index=hexagons.index, crs=hexagons.crs)

    # =============================================================================
    # Post geopandas V 0.1.0, the naturalearth_lowres dataset is no longer available.
    # Natural earth make them available via a github repository: https://github.com/nvkelso/natural-earth-vector/tree/master
    # =============================================================================
    # warnings.filterwarnings('ignore', category=FutureWarning)
    # world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')) # may need to switch to higher res
    # countries = world.drop(columns=['pop_est', 'continent', 'iso_a3', 'gdp_md_est'])
    
    world = gpd.read_file('/vsicurl/https://github.com/nvkelso/natural-earth-vector/raw/master/10m_cultural/ne_10m_admin_0_countries.shp')
    countries = world[["NAME", 'geometry']]
    countries = countries.rename(columns={'NAME':'country'})
    
    if how == "centroid_within":
        # risk that a hexagon is excluded if centroid lies directly on country boundary
        hexagons["country"] = gpd.sjoin(hexagon_centroids, countries, predicate='within').country
    elif how == "centroid_intersect":
        # risk of duplicated hexagons if centroid lies directly on country boundary
        hexagons["country"] = gpd.sjoin(hexagon_centroids, countries, predicate='intersects').country 
    elif how == "area_majority":
        raise NotImplementedError(f"{how} not implemented yet")
    elif how == "hexagon_within":
        hexagons["country"] = gpd.sjoin(hexagons, countries, predicate='within').country 
    else:
        raise NotImplementedError(f"{how} not implemented yet")
    
    hexagons.to_file(outputpath, driver='GeoJSON')
    
    hexagons['country'] = hexagons['country'].apply(lambda x: x if x in country_parameters.index.values else 'Other')
    
    return hexagons, hexagons[hexagons["country"]==country_name_clean]
    

def load_hexagons_temporary(country_parameters, 
                            filepath='Data/hexagons_with_country.geojson'):
    # Handle any hexagons at edges in the geojson which are labelled with a country we aren't analyzing

    # Read the GeoJSON file
    with open(filepath, 'r') as file:
        data = json.load(file)

    # If the country of any hexagon is not in the country_parameters file, set the country to "Other" instead
    for feature in data['features']:
        # Access and modify properties
        if not feature['properties']['country'] in list(country_parameters.index.values):
            feature['properties']['country'] = "Other"

    # Write the modified GeoJSON back to the file
    with open('Data/hexagons_with_country.geojson', 'w') as file:
        json.dump(data, file)

    # Now, load the Hexagon file in geopandas
    hexagon = gpd.read_file('Data/hexagons_with_country.geojson')
    return hexagon

#%% distance functions
def geodesic_matrix(gdf1, gdf2, desc="calculating distances"):
    distances = np.empty((gdf1.shape[0], gdf2.shape[0]))
    warnings.filterwarnings('ignore', category=UserWarning)
    with tqdm(total=gdf1.shape[0] * gdf2.shape[0], desc=desc) as pbar:
        # the use of centroid throws a UserWarning when a geographic CRS (e.g. EPSG=4326) is used.
        # ideally you should convert to a projected crs (ideally with equal area cylindrical?) then back again.
        for i, p1 in enumerate(gdf1.centroid):
            for j, p2 in enumerate(gdf2.centroid):
                distances[i,j] = geopy.distance.geodesic((p1.y, p1.x),
                                                         (p2.y, p2.x)).km
                pbar.update()
    warnings.filterwarnings('default', category=UserWarning)
    return pd.DataFrame(distances, index=gdf1.index, columns=gdf2.index)

def find_nearest_hex(idx, hex_to_X_distance):
    hix = hex_to_X_distance.loc[:, idx].sort_values().index[0] # index of the nearest hexagon to the demand centre (contain might not work at low hexagon resolution)
    return hix

#%% economic functions
def CRF(interest,lifetime):
    '''
    Calculates the capital recovery factor of a capital investment.

    Parameters
    ----------
    interest : float
        interest rate.
    lifetime : float or integer
        lifetime of asset.

    Returns
    -------
    CRF : float
        present value factor.

    '''
    interest = float(interest)
    lifetime = float(lifetime)

    CRF = (((1+interest)**lifetime)*interest)/(((1+interest)**lifetime)-1)
    return CRF

#%% road and trucking 
def calculate_road_construction(hexagon, infra_data, country_parameters):
    road_capex_long = infra_data.at['Long road','CAPEX (euros/km)']            #â¬/km from John Hine, converted to Euro (Assumed earth to paved road)
    road_capex_short = infra_data.at['Short road','CAPEX (euros/km)']         #â¬/km for raods < 10 km, from John Hine, converted to Euro (Assumed earth to paved road)
    road_opex = infra_data.at['Short road','OPEX (euros/km/year)']                 #â¬/km/year from John Hine, converted to Euro (Assumed earth to paved road)

    if hexagon['road_dist']==0:
        road_construction_costs = 0.
    elif hexagon['road_dist']!=0 and hexagon['road_dist']<10:
        road_construction_costs = (hexagon['road_dist']
                                   * road_capex_short
                                   * CRF(country_parameters.loc[hexagon['country'],'Infrastructure interest rate'],
                                         country_parameters.loc[hexagon['country'],'Infrastructure lifetime (years)'])
                                   + hexagon['road_dist'] * road_opex)
    else:
        road_construction_costs = (hexagon['road_dist'] 
                                   * road_capex_long 
                                   * CRF(country_parameters.loc[hexagon['country'], 'Infrastructure interest rate'],
                                         country_parameters.loc[hexagon['country'],'Infrastructure lifetime (years)'])
                                   + hexagon['road_dist'] * road_opex)
    return road_construction_costs

def mineral_trucking_costs(transport_state, distance, quantity, interest, excel_path= "Parameters/transport_parameters.xlsx"):
    '''
    calculates the annual cost of transporting resource by truck.

    Parameters
    ----------
    transport_state : string
        state resource is transported in.
    distance : float
        distance between production site and demand site.
    quantity : float
        annual amount of resource to transport.
    interest : float
        interest rate on capital investments.
    excel_path : string
        path to transport_parameters.xlsx file
        
    Returns
    -------
    annual_costs : float
        annual cost of hydrogen transport with specified method.
    '''
    daily_quantity = quantity / 365

    transport_parameters = pd.read_excel(excel_path,
                                         sheet_name = transport_state,
                                         index_col = 'Parameter'
                                         ).squeeze('columns')

    average_truck_speed = transport_parameters['Average truck speed (km/h)']                #km/h
    working_hours = transport_parameters['Working hours (h/day)']                     #h/day
    diesel_price = transport_parameters['Diesel price (euros/L)']                    #€/l
    costs_for_driver = transport_parameters['Costs for driver (euros/h)']                  #€/h
    working_days = transport_parameters['Working days (per year)']                      #per year
    max_driving_dist = transport_parameters['Max driving distance (km/a)']               #km/a Maximum driving distance per truck per year

    spec_capex_truck = transport_parameters['Spec capex truck (euros)']               #€
    spec_opex_truck = transport_parameters['Spec opex truck (% of capex/a)']                  #% of CAPEX/a
    diesel_consumption = transport_parameters['Diesel consumption (L/100 km)']                 #l/100km
    truck_lifetime = transport_parameters['Truck lifetime (a)']                      #a

    spec_capex_trailor = transport_parameters['Spec capex trailer (euros)']
    spec_opex_trailor =transport_parameters['Spec opex trailer (% of capex/a)']
    net_capacity = transport_parameters['Net capacity (kg)']                     #kg
    trailor_lifetime = transport_parameters['Trailer lifetime (a)']                   #a
    loading_unloading_time = transport_parameters['Loading unloading time (h)']            #hours 


    # max_day_dist = max_driving_dist/working_days
    amount_deliveries_needed = daily_quantity / net_capacity
    deliveries_per_truck = working_hours / (loading_unloading_time + (2 * distance / average_truck_speed))
    trailors_needed = round((amount_deliveries_needed / deliveries_per_truck) + 0.5, 0)
    total_drives_day = round(amount_deliveries_needed + 0.5, 0) # not in ammonia calculation
    if transport_state == 'NH3': #!!! double checking if this is needed with Leander
        trucks_needed = trailors_needed
    else:
        trucks_needed = max(round((total_drives_day * 2 * distance * working_days / max_driving_dist) + 0.5, 0), trailors_needed)

    capex_trucks = trucks_needed * spec_capex_truck
    capex_trailor = trailors_needed * spec_capex_trailor
    # capex_total = capex_trailor + capex_trucks
    # this if statement seems suspect to me-- how can a fractional number of deliveries be completed?
    if amount_deliveries_needed < 1:
        fuel_costs = (amount_deliveries_needed * 2 * distance * 365 / 100) * diesel_consumption * diesel_price
        wages = amount_deliveries_needed * ((distance / average_truck_speed) * 2 + loading_unloading_time) * working_days * costs_for_driver
    
    else:
        fuel_costs = (round(amount_deliveries_needed + 0.5) * 2 * distance * 365 / 100) * diesel_consumption * diesel_price
        wages = round(amount_deliveries_needed + 0.5) * ((distance / average_truck_speed) * 2 + loading_unloading_time) * working_days * costs_for_driver

    annual_costs = ((capex_trucks * CRF(interest, truck_lifetime) + capex_trailor * CRF(interest, trailor_lifetime))
                    + capex_trucks * spec_opex_truck 
                    + capex_trailor * spec_opex_trailor 
                    + fuel_costs 
                    + wages)
    
    costs_per_unit = annual_costs / quantity
    return costs_per_unit, transport_state

#%% rail and train
def calculate_rail_construction(hexagon, infra_data, country_parameters):
    rail_capex_long = infra_data.at['Long rail', 'CAPEX (euros/km)']  # euros / km, assuming earth to rail
    rail_capex_short = infra_data.at['Short rail','CAPEX (euros/km)']  # euros / km, assuming earth to rail
    rail_opex = infra_data.at['Short rail','OPEX (euros/km/year)']  # euros/km/year
    
    if hexagon['rail_dist']==0:
        rail_construction_costs = 0.
    elif hexagon['rail_dist']!=0 and hexagon['rail_dist']<10:
        rail_construction_costs = (hexagon['rail_dist']
                                   * rail_capex_short
                                   * CRF(country_parameters.loc[hexagon['country'],'Infrastructure interest rate'],
                                         country_parameters.loc[hexagon['country'],'Infrastructure lifetime (years)'])
                                   + hexagon['rail_dist'] * rail_opex)
    else:
        rail_construction_costs = (hexagon['rail_dist'] 
                                   * rail_capex_long 
                                   * CRF(country_parameters.loc[hexagon['country'], 'Infrastructure interest rate'],
                                         country_parameters.loc[hexagon['country'],'Infrastructure lifetime (years)'])
                                   + hexagon['rail_dist'] * rail_opex)
    return rail_construction_costs
    
def mineral_train_costs(transport_state, distance, quantity, interest, excel_path= "Parameters/rail_parameters.xlsx"):
    '''
    Estimates the annual cost of transporting resource by train.
    Method:
        1. 
        2.
    Assumptions:
        1.
        2.

    Parameters
    ----------
    transport_state : string
        state resource is transported in.
    distance : float
        distance between production site and demand site.
    quantity : float
        annual amount of resource to transport.
    interest : float
        interest rate on capital investments.
    excel_path : string
        path to transport_parameters.xlsx file
        
    Returns
    -------
    annual_costs : float
        annual cost of hydrogen transport with specified method.
    '''
    daily_quantity = quantity / 365

    transport_parameters = pd.read_excel(excel_path,
                                         sheet_name = transport_state,
                                         index_col = 'Parameter'
                                         ).squeeze('columns')

    average_train_speed = transport_parameters['Average train speed (km/h)']
    working_hours = transport_parameters['Working hours (h/day)']
    diesel_price = transport_parameters['Diesel price (euros/L)']
    costs_for_driver = transport_parameters['Costs for driver (euros/h)']
    working_days = transport_parameters['Working days (per year)']
    max_driving_dist = transport_parameters['Max distance (km/a)']

    spec_capex_loco = transport_parameters['Spec capex loco (euros)']
    spec_opex_loco = transport_parameters['Spec opex loco (% of capex/a)']
    diesel_consumption = transport_parameters['Diesel consumption (L/100 km)']
    loco_lifetime = transport_parameters['Loco lifetime (a)']

    spec_capex_wagon = transport_parameters['Spec capex wagon (euros)']
    spec_opex_wagon =transport_parameters['Spec opex wagon (% of capex/a)']
    net_capacity = transport_parameters['Net capacity (kg)']
    wagon_lifetime = transport_parameters['Wagon lifetime (a)']
    loading_unloading_time = transport_parameters['Loading unloading time (h)']
    max_wagons = transport_parameters['Max wagons per loco']

    # max_day_dist = max_driving_dist/working_days
    
    # calc minimum journeys based on  max # of wagons
    # wagon_journeys_needed = np.ceil(quantity/(net_capacity * max_wagons))
    
    max_journeys_per_loco = (working_hours * working_days) / (loading_unloading_time + (2 * distance / average_train_speed))  # max journeys a single loco can make per year
    max_quantity_per_train = max_journeys_per_loco * net_capacity * max_wagons  # max quantity a single train (loco + max wagons) can transfer per year
    min_locos = np.ceil(quantity/max_quantity_per_train)  # minimum number of locos required assuming 1 loco per train
    min_wagons_per_train = np.ceil((quantity/min_locos)/(net_capacity))  # minimum wagons needed per train
    
    total_train_journeys = quantity / (min_wagons_per_train * net_capacity)
    
    # # calc min number of wagons per train based on min journeys
    # wagons_per_train = np.ceil(quantity/net_capacity)/min_journeys_needed
    
    # deliveries_per_train = working_hours / (loading_unloading_time + (2 * distance / average_train_speed)) # per day
    
    # trailors_needed = round((amount_deliveries_needed / deliveries_per_truck) + 0.5, 0)
    # total_drives_day = round(amount_deliveries_needed + 0.5, 0) # not in ammonia calculation
    # if transport_state == 'NH3': #!!! double checking if this is needed with Leander
    #     trucks_needed = trailors_needed
    # else:
    #     trucks_needed = max(round((total_drives_day * 2 * distance * working_days / max_driving_dist) + 0.5, 0), trailors_needed)

    capex_loco = min_locos * spec_capex_loco
    capex_wagon = min_wagons_per_train * spec_capex_wagon
    # capex_total = capex_trailor + capex_trucks
    # this if statement seems suspect to me-- how can a fractional number of deliveries be completed?
    
    fuel_costs = ((total_train_journeys * 2 * distance) / 100) * diesel_consumption * diesel_price
    wages = total_train_journeys * ((distance / average_train_speed) * 2 + loading_unloading_time) * costs_for_driver
    
    # if amount_deliveries_needed < 1:
    #     fuel_costs = (amount_deliveries_needed * 2 * distance * 365 / 100) * diesel_consumption * diesel_price
    #     wages = amount_deliveries_needed * ((distance / average_truck_speed) * 2 + loading_unloading_time) * working_days * costs_for_driver
    
    # else:
    #     fuel_costs = (round(amount_deliveries_needed + 0.5) * 2 * distance * 365 / 100) * diesel_consumption * diesel_price
    #     wages = round(amount_deliveries_needed + 0.5) * ((distance / average_truck_speed) * 2 + loading_unloading_time) * working_days * costs_for_driver

    annual_costs = ((capex_loco * CRF(interest, loco_lifetime) + capex_wagon * CRF(interest, wagon_lifetime))
                    + capex_loco * spec_opex_loco 
                    + capex_wagon * spec_opex_wagon 
                    + fuel_costs 
                    + wages)
    
    costs_per_unit = annual_costs / quantity
    return costs_per_unit, transport_state






# def cheapest_mineral_trucking_strategy(final_state, quantity, distance, 
#                                 elec_costs, diesel_costs, heat_costs,
#                                 interest, 
#                                 elec_costs_demand, elec_cost_grid = 0.):
#     '''
#     Direct patch to trucking costs as for mineral X, studying the conversion.

#     Parameters
#     ----------
#     final_state : string
#         final state for hydrogen demand.
#     quantity : float
#         annual demand for hydrogen in kg.
#     distance : float
#         distance to transport hydrogen.
#     elec_costs : float
#         cost per kWh of electricity at hydrogen production site.
#     heat_costs : float
#         cost per kWh of heat.
#     interest : float
#         interest on conversion and trucking capital investments (not including roads).
#     elec_costs_demand : float
#         cost per kWh of electricity at hydrogen demand site.
#     elec_cost_grid : float
#         grid electricity costs that pipeline compressors pay. Default 0.
    
#     Returns
#     -------
#     costs_per_unit : float
#         storage, conversion, and transport costs for the cheapest trucking option.
#     cheapest_option : string
#         the lowest-cost state in which to transport hydrogen by truck.

#     '''
    
#     if final_state == "CuAnode":
#         # convert to CuAnode then truck
#         at = (mineral_conversion_stand(final_state, quantity, elec_costs,
#                                       diesel_costs, heat_costs, interest)[2]
#               + trucking_costs('CuAnode', distance, quantity, interest,
#                                transport_excel_path))
#         # truck then convert to CuAnode
#         ta = (trucking_costs('CuConcentrate', distance, quantity, interest,
#                          transport_excel_path)
#               + mineral_conversion_stand(final_state, quantity, elec_costs_demand,
#                                             diesel_costs, heat_costs, interest)[2])
    
#         lowest_cost = np.nanmin([at, ta])
        
#         if at == lowest_cost:
#             cheapest_option = 'CuAnode'
#         elif ta == lowest_cost:
#             cheapest_option = 'CuConcentrate'

#         costs_per_unit = lowest_cost / quantity
        
#         return costs_per_unit, cheapest_option
        
        
#     elif final_state == "CuCathode":
#        # convert to CuAnode then truck
#        ct = (mineral_conversion_stand(final_state, quantity, elec_costs,
#                                      diesel_costs, heat_costs, interest)[2]
#              + trucking_costs('CuCathode', distance, quantity, interest,
#                               transport_excel_path))
#        # truck then convert to CuAnode
#        tc = (trucking_costs('CuConcentrate', distance, quantity, interest,
#                         transport_excel_path)
#              + mineral_conversion_stand(final_state, quantity, elec_costs_demand,
#                                            diesel_costs, heat_costs, interest)[2])
   
#        lowest_cost = np.nanmin([ct, tc])
       
#        if ct == lowest_cost:
#            cheapest_option = 'CuCathode'
#        elif tc == lowest_cost:
#            cheapest_option = 'CuConcentrate'

#        costs_per_unit = lowest_cost / quantity
       
#        return costs_per_unit, cheapest_option
   
#     elif final_state == 'CuConcentrate':
#         lowest_cost = trucking_costs('CuConcentrate', distance, quantity, interest,
#                          transport_excel_path)
#         cheapest_option = 'CuConcentrate'
#         costs_per_unit = lowest_cost / quantity
#         return costs_per_unit, cheapest_option
#     else:
#         raise NotImplementedError(f'Conversion costs for {final_state} not currently supported.')

#%% demand schedules
def demand_schedule(quantity, transport_state,
                    transport_excel_path="Parameters/transport_parameters.xlsx",
                    weather_excel_path="Parameters/weather_parameters.xlsx"):
    '''
    calculates hourly product demand for truck shipment and train transport.

    Parameters
    ----------
    quantity : float
        annual amount of hydrogen to transport in kilograms.
    transport_state : string
        product state
    transport_excel_path : string
        path to transport_parameters.xlsx file
    weather_excel_path : string
        path to transport_parameters.xlsx file
            
    Returns
    -------
    trucking_hourly_demand_schedule : pandas DataFrame
        hourly demand profile for hydrogen trucking.
    pipeline_hourly_demand_schedule : pandas DataFrame
        hourly demand profile for pipeline transport.
    '''
    transport_parameters = pd.read_excel(transport_excel_path,
                                         sheet_name = transport_state,
                                         index_col = 'Parameter'
                                         ).squeeze('columns')
    weather_parameters = pd.read_excel(weather_excel_path,
                                       index_col = 'Parameters',
                                       ).squeeze('columns')
    truck_capacity = transport_parameters['Net capacity (kg)']
    start_date = weather_parameters['Start date']
    end_date = weather_parameters['End date (not inclusive)']

    # schedule for trucking
    annual_deliveries = quantity / truck_capacity
    quantity_per_delivery = quantity / annual_deliveries
    index = pd.date_range(start_date, end_date, periods=annual_deliveries)
    trucking_demand_schedule = pd.DataFrame(quantity_per_delivery, index=index, columns = ['Demand'])
    return trucking_demand_schedule
    trucking_hourly_demand_schedule = trucking_demand_schedule.resample('H').sum().fillna(0.)

    # # schedule for pipeline
    # index = pd.date_range(start_date, end_date, freq = 'H')
    # pipeline_hourly_quantity = quantity/index.size
    # pipeline_hourly_demand_schedule = pd.DataFrame(pipeline_hourly_quantity, index=index,  columns = ['Demand'])

    return trucking_hourly_demand_schedule


def demand_schedule_constant(quantity, transport_state,
                    transport_excel_path="Parameters/transport_parameters.xlsx",
                    weather_excel_path="Parameters/weather_parameters.xlsx"):
    '''
    calculates  a constant hourly product demand if storage between production and shipping not important

   
    '''
    # transport_parameters = pd.read_excel(transport_excel_path,
    #                                      sheet_name = transport_state,
    #                                      index_col = 'Parameter'
    #                                      ).squeeze('columns')
    weather_parameters = pd.read_excel(weather_excel_path,
                                       index_col = 'Parameters',
                                       ).squeeze('columns')
    # truck_capacity = transport_parameters['Net capacity (kg)']
    start_date = weather_parameters['Start date']
    end_date = weather_parameters['End date (not inclusive)']

    # schedule for trucking
    # annual_deliveries = quantity / truck_capacity
    # quantity_per_delivery = quantity / annual_deliveries
    
    quantity_per_hour = quantity / (365*24)
    index = pd.date_range(start_date, end_date, freq="h")
    trucking_hourly_demand_schedule = pd.DataFrame(quantity_per_hour, index=index, columns = ['Demand'])

    # # schedule for pipeline
    # index = pd.date_range(start_date, end_date, freq = 'H')
    # pipeline_hourly_quantity = quantity/index.size
    # pipeline_hourly_demand_schedule = pd.DataFrame(pipeline_hourly_quantity, index=index,  columns = ['Demand'])

    return trucking_hourly_demand_schedule

#%% grid connection and construction
def calculate_grid_construction(hexagon, infra_data, country_parameters):
    grid_capex = infra_data.at['Grid','CAPEX (euros/km)']
    grid_opex = infra_data.at['Grid','OPEX (euros/km/year)']                

    if hexagon['grid_dist']==0:
        grid_construction_costs = 0.
    else:
        grid_construction_costs = (hexagon['grid_dist'] 
                                   * grid_capex 
                                   * CRF(country_parameters.loc[hexagon['country'], 'Infrastructure interest rate'],
                                         country_parameters.loc[hexagon['country'],'Infrastructure lifetime (years)'])
                                   + hexagon['grid_dist'] * grid_opex)
    return grid_construction_costs
    

#%% energy system optimisation
def optimize_facility(wind_potential, pv_potential, times, demand_profile,
                      demand_state, elec_kWh_per_kg, wind_max_capacity, pv_max_capacity, 
                            country_series):
    '''
   Optimizes the size of green processing facility components based on renewable potential and country parameters. 

    

    '''


    # Set up network
    # Import a generic network
    n = pypsa.Network(override_component_attrs=aux.create_override_components())

    # Set the time values for the network
    n.set_snapshots(times)

    # Import the design of the H2 plant into the network
    n.import_from_csv_folder("Parameters/Basic_MinX_plant")

    # Import demand profile
    # Note: All flows are in MW or MWh, conversions for Concentrate done using 0.717 kWh per kg

    n.add('Load',
          f'{demand_state} demand',
          bus = 'Power',
          p_set = (demand_profile['Demand'] * elec_kWh_per_kg) / 1000,
          )

    # Send the weather data to the model
    n.generators_t.p_max_pu['Wind'] = wind_potential
    n.generators_t.p_max_pu['Solar'] = pv_potential

    # specify maximum capacity based on land use
    n.generators.loc['Wind','p_nom_max'] = wind_max_capacity
    n.generators.loc['Solar','p_nom_max'] = pv_max_capacity

    # specify technology-specific and country-specific WACC and lifetime here
    n.generators.loc['Wind','capital_cost'] = n.generators.loc['Wind','capital_cost']\
        * CRF(country_series['Wind interest rate'], country_series['Wind lifetime (years)'])
    n.generators.loc['Solar','capital_cost'] = n.generators.loc['Solar','capital_cost']\
        * CRF(country_series['Solar interest rate'], country_series['Solar lifetime (years)'])
    for item in [n.links, n.stores, n.storage_units]:
        item.capital_cost = item.capital_cost * CRF(country_series['Plant interest rate'], country_series['Plant lifetime (years)'])
        
    # Solve the model
    solver = 'gurobi'
    with suppress_output():
        n.lopf(solver_name=solver,
               solver_options = {'OutputFlag': 0},
               pyomo=False,
               extra_functionality=aux.extra_functionalities,
               )
    # Output results

    lcom = n.objective/(n.loads_t.p_set.sum()[0] / elec_kWh_per_kg * 1000) # convert back to kg 
    wind_capacity = n.generators.p_nom_opt['Wind']
    solar_capacity = n.generators.p_nom_opt['Solar']
    # electrolyzer_capacity = np.nan # n.links.p_nom_opt['Electrolysis']
    battery_capacity = n.storage_units.p_nom_opt['Battery']
    # h2_storage = np.nan # n.stores.e_nom_opt['Compressed H2 Store']
    # print(lcom)
    
    if math.isnan(lcom):
        # print("nan wind len: {}".format(wind_potential.shape))
        # print("nan demand len: {}".format(demand_profile.shape))
        pass
        # print("demand: {}".format(demand_profile['Demand'].sum()),
        #       "pv lim: {}".format(pv_max_capacity),
        #       "pv CF: {}".format(pv_potential.mean().item()),
        #       "wind lim: {}".format(wind_max_capacity),
        #       "wind CF: {}".format(wind_potential.mean().item()),
        #       )
        # print("Annual demand (MWh) {}".format((demand_profile['Demand'].sum() * elec_kWh_per_kg) / 1000))
        # print("Annual solar (MWh) {}".format(24*365*pv_max_capacity*pv_potential.mean().item()))
        # print("Annual wind (MWh) {}".format(24*365*wind_max_capacity*wind_potential.mean().item()))
    else:
        print("success wind len: {}".format(wind_potential.shape))
        print("success demand len: {}".format(demand_profile.shape))
    return lcom, wind_capacity, solar_capacity, battery_capacity, n


def optimize_offgrid_facility(wind_potential, pv_potential, times, demand_profile,
                      demand_state, elec_kWh_per_kg, wind_max_capacity, pv_max_capacity, 
                            country_series):
    '''
   Optimizes the size of green processing facility components based on renewable potential and country parameters. 

    

    '''
    
    solar = pv_potential.to_series().loc[:demand_profile.index[-1]]
    wind = wind_potential.to_series().loc[:demand_profile.index[-1]]
    load = demand_profile["Demand"]
    
    
    generator_params = pd.read_csv("Parameters/Basic_MinX_plant/generators.csv",
                                   index_col=0)
    storage_params = pd.read_csv("Parameters/Basic_MinX_plant/storage_units.csv",
                                   index_col=0)
    # there is a future warning from pandas within pypsa components re. setting an item of incompatible dtype.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        # Set up network
        # Import a generic network
        n = pypsa.Network()
    
        # Set the time values for the network
        n.set_snapshots(demand_profile.index)
    
        # Add buses
        n.add("Bus", "AC_bus")
        n.add("Bus", "DC_bus")
        
        # Import demand profile
        # Note: All flows are in MW or MWh, conversions for Concentrate done using 0.717 kWh per kg
        
        # add the load
        n.add('Load',
              f'{demand_state} demand',
              bus = 'AC_bus',
              p_set = (load * elec_kWh_per_kg) / 1000,
              )
        
        # add generation
        n.add("Generator",
                "solar",
                bus="DC_bus",
                p_nom_extendable=True,
                carrier="solar",
                p_max_pu=solar,
                p_nom_max=pv_max_capacity)
    
        n.add("Generator",
                "wind",
                bus="DC_bus",
                p_nom_extendable=True,
                carrier="wind",
                p_max_pu=wind,
                p_nom_max=wind_max_capacity)
    
        # Add battery storage
        n.add("StorageUnit",
                "battery",
                bus="DC_bus",
                p_nom_extendable=True,
                cyclic_state_of_charge=True,
                state_of_charge_initial=0.5,
                max_hours=4)
        
        # Add converters
        n.add("Link",
                "inverter",
                bus0="DC_bus",
                bus1="AC_bus",
                p_nom_extendable=True,
                efficiency=0.95)
        
        n.add("Link",
                "rectifier",
                bus0="AC_bus",
                bus1="DC_bus",
                p_nom_extendable=True,
                efficiency=0.95)  # Example efficiency
    
        # Set costs
        n.generators.loc["solar", "capital_cost"] = (generator_params.loc["Solar", "capital_cost"]
                                                     * CRF(country_series['Solar interest rate'], country_series['Solar lifetime (years)']))
        n.generators.loc["wind", "capital_cost"] = (generator_params.loc["Wind", "capital_cost"]
                                                     * CRF(country_series['Wind interest rate'], country_series['Wind lifetime (years)']))
        n.storage_units.loc["battery", "capital_cost"] = (storage_params.loc["Battery", "capital_cost"]
                                                          * CRF(country_series['Solar interest rate'], country_series['Solar lifetime (years)']))
        n.links.loc["inverter", "capital_cost"] = (50
                                                     * CRF(country_series['Plant interest rate'], country_series['Plant lifetime (years)']))
        n.links.loc["rectifier", "capital_cost"] = (50
                                                 * CRF(country_series['Plant interest rate'], country_series['Plant lifetime (years)']))
    
        # Solve the model
        solver = 'gurobi'
        with suppress_output():
            n.lopf(n.snapshots,
                   solver_name=solver,
                   solver_options = {'OutputFlag': 0},
                   pyomo=False,
                   )

    # Output results
    lcom = n.objective/(n.loads_t.p_set.sum().iloc[0] / elec_kWh_per_kg * 1000) # convert back to kg 
    wind_capacity = n.generators.p_nom_opt['wind']
    solar_capacity = n.generators.p_nom_opt['solar']
    battery_capacity = n.storage_units.p_nom_opt['battery']
    # print(lcom)
    
    return lcom, wind_capacity, solar_capacity, battery_capacity, n


def optimize_ES_gridconnected(wind_potential, pv_potential, times,
                              demand_profile, demand_state,
                              elec_kWh_per_kg, wind_max_capacity,
                              pv_max_capacity, country_series, battery_p_max=None):
    '''
   Optimizes the size of green processing facility components based on renewable potential and country parameters. 

    

    '''
    
    solar = pv_potential.to_series().loc[:demand_profile.index[-1]]
    wind = wind_potential.to_series().loc[:demand_profile.index[-1]]
    load = demand_profile["Demand"]
    
    
    generator_params = pd.read_csv("Parameters/Basic_MinX_plant/generators.csv",
                                   index_col=0)
    storage_params = pd.read_csv("Parameters/Basic_MinX_plant/storage_units.csv",
                                   index_col=0)
    # there is a future warning from pandas within pypsa components re. setting an item of incompatible dtype.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        # Set up network
        # Import a generic network
        n = pypsa.Network()
    
        # Set the time values for the network
        n.set_snapshots(demand_profile.index)
    
        # Add buses
        n.add("Bus", "AC_bus")
        n.add("Bus", "DC_bus")
        
        # Import demand profile
        # Note: All flows are in MW or MWh, conversions for Concentrate done using elec_kWh_per_kg parameter
        
        # add the load
        n.add('Load',
              f'{demand_state} demand',
              bus = 'AC_bus',
              p_set = (load * elec_kWh_per_kg) / 1000,
              )
        
        if pv_max_capacity>0:
            # add generation
            n.add("Generator",
                    "solar",
                    bus="DC_bus",
                    p_nom_extendable=True,
                    carrier="solar",
                    p_max_pu=solar,
                    p_nom_max=pv_max_capacity,
                    capital_cost=(generator_params.loc["Solar", "capital_cost"]
                                  * CRF(country_series['Solar interest rate'],
                                        country_series['Solar lifetime (years)'])))
        if wind_max_capacity>0:
            n.add("Generator",
                    "wind",
                    bus="DC_bus",
                    p_nom_extendable=True,
                    carrier="wind",
                    p_max_pu=wind,
                    p_nom_max=wind_max_capacity,
                    capital_cost=(generator_params.loc["Wind", "capital_cost"]
                                  * CRF(country_series['Wind interest rate'],
                                        country_series['Wind lifetime (years)'])))
    
        if isinstance(battery_p_max, type(None)):
            # Add battery storage
            n.add("StorageUnit",
                    "battery",
                    bus="DC_bus",
                    p_nom_extendable=True,
                    cyclic_state_of_charge=True,
                    state_of_charge_initial=0.5,
                    max_hours=4,
                    capital_cost=(storage_params.loc["Battery", "capital_cost"]
                                  * CRF(country_series['Solar interest rate'],
                                        country_series['Solar lifetime (years)'])))
        elif battery_p_max==0:
            # do not add battery
            pass
        else:
            n.add("StorageUnit",
                    "battery",
                    bus="DC_bus",
                    p_nom_extendable=True,
                    cyclic_state_of_charge=True,
                    state_of_charge_initial=0.5,
                    max_hours=4,
                    p_nom_max=battery_p_max)
            
        n.add("Generator",
              "grid",
              bus="AC_bus",
              p_nom_extendable=True,
              marginal_cost=country_series["Electricity price (euros/kWh)"] * 1000,
              capital_cost=(country_series["Grid connection cost (euros/kW)"] * 1000
                            * CRF(country_series['Plant interest rate'], country_series['Plant lifetime (years)'])))
        
        # Add converters
        n.add("Link",
                "inverter",
                bus0="DC_bus",
                bus1="AC_bus",
                p_nom_extendable=True,
                efficiency=0.95,
                capital_cost=(50
                              * CRF(country_series['Plant interest rate'],
                                    country_series['Plant lifetime (years)'])))
        
        n.add("Link",
                "rectifier",
                bus0="AC_bus",
                bus1="DC_bus",
                p_nom_extendable=True,
                efficiency=0.95,
                capital_cost=(50
                              * CRF(country_series['Plant interest rate'],
                                    country_series['Plant lifetime (years)'])))  # Example efficiency
    
        
        # Solve the model
        solver = 'gurobi'
        with suppress_output():
            n.lopf(n.snapshots,
                   solver_name=solver,
                   solver_options = {'OutputFlag': 0},
                   pyomo=False,
                   )
    
    


# Output results
    lcom = n.objective/(n.loads_t.p_set.sum().iloc[0] / elec_kWh_per_kg * 1000) # convert back to kg 
    if wind_max_capacity>0:
        wind_capacity = n.generators.p_nom_opt['wind']
    else:
        wind_capacity = np.nan
    if pv_max_capacity>0:
        solar_capacity = n.generators.p_nom_opt['solar']
    else:
        solar_capacity = np.nan
    if isinstance(battery_p_max, type(None)) or battery_p_max>0:
        battery_capacity = n.storage_units.p_nom_opt['battery']
    else:
        battery_capacity = np.nan
    grid_capacity = n.generators.p_nom_opt["grid"]
    # print(lcom)
    
    return lcom, wind_capacity, solar_capacity, battery_capacity, grid_capacity, n

#%% facility construction
def mineral_conversion_stand(final_state, quantity, interest,
                             conversion_excel_path="Parameters/conversion_parameters.xlsx"):
    '''
    calculates the annual cost of facility (excluding energy) 
    for converting mineral concentrate to a given state
    '''
    
    daily_throughput = quantity/365
    
    conversion_parameters = pd.read_excel(conversion_excel_path,
                                         sheet_name = final_state,
                                         index_col = 'Parameter'
                                         ).squeeze('columns')

    if final_state == 'CuConcentrate':
        annual_costs = 0 
        return annual_costs

    else:
        capex_quadratic_coefficient = conversion_parameters['Capex quadratic coefficient (euros (kg product)-2)']
        capex_linear_coefficient = conversion_parameters['Capex linear coefficient (euros per kg product)']
        capex_constant = conversion_parameters['Capex constant (euros)']
        opex_plant = conversion_parameters['Opex (% of capex)']
        plant_lifetime = conversion_parameters['Plant lifetime (a)']
        
       
        
        capex_plant = (capex_quadratic_coefficient *(daily_throughput**2)
                       + capex_linear_coefficient * daily_throughput
                       + capex_constant)

        annual_costs = ((capex_plant * CRF(interest, plant_lifetime))
                        + (capex_plant * opex_plant))
        return annual_costs / quantity
