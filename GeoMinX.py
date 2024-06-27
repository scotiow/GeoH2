# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:15:32 2024

@author: Scot Wheeler
"""

import atlite
import geopandas as gpd
import pypsa
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import p_H2_aux as aux
import GeoMinX_functions as gmx
from functions import CRF
import numpy as np
import logging
import time
from tqdm.auto import tqdm
import os
from shapely.geometry import Point
import geopy
import warnings
from shapely.errors import ShapelyDeprecationWarning
# this ignores the following warning when using atlite: 
    # ShapelyDeprecationWarning: STRtree will be changed in 2.0.0 and will not be compatible with versions < 2.
# be aware, this may lead to a future error if using newer versions of Shapely
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

logging.basicConfig(level=logging.ERROR)

demand_states = ['CuAnode','CuCathode','CuConcentrate']

# Create Resources folder to save results if it doesn't already exist
if not os.path.exists('Resources'):
    os.makedirs('Resources')

if __name__=='__main__':
    
    
    # load parameter files
    (infra_data, global_data, mineral_data, demand_center_list,
            generation_center_list, country_parameters) = gmx.import_parameter_files()
    conversion_parameters = gmx.import_conversion_parameters(demand_states) # from feedstock
    
    road_construction = global_data['Road construction allowed']
    rail_construction = global_data['Rail construction allowed']
    
    #%% load spider data
    pass

    #%% load glaes data
    pass
    
    #%% merge data - previously GeoH2_data_prep
    # temporary just import the prepared data as per GeoH2 transport and conversion
    hexagons_gdf = gmx.load_hexagons_temporary(country_parameters)
    num_hex = hexagons_gdf.shape[0]
    
    demand_points_gdf = gpd.GeoDataFrame(demand_center_list, geometry=[Point(xy) for xy in zip(demand_center_list['Lon [deg]'], demand_center_list['Lat [deg]'])]).set_crs(epsg=4326)
    num_dem = demand_points_gdf.shape[0]

    feedstock_points_gdf = gpd.GeoDataFrame(generation_center_list, geometry=[Point(xy) for xy in zip(generation_center_list['Lon [deg]'], generation_center_list['Lat [deg]'])]).set_crs(epsg=4326)
    num_gen = feedstock_points_gdf.shape[0]
    
    #%% import weather data, pv and wind profiles
    pv_profile, wind_profile = gmx.get_pv_wind_profiles(hexagons_gdf)
    
    #%% calculate distances
# =============================================================================
#     Would it be a good idea to calculate the transport costs for every 
#     combination of mines, hexagons, and demands first - or will this be too
#     high a memory load? Opportunity to skip calculations where not needed.
# =============================================================================
    
    hexagon_to_demand_distance_matrix = gmx.geodesic_matrix(hexagons_gdf,
                                                     demand_points_gdf,
                                                     desc="Calculating distance to demand")
    hexagon_to_feedstock_distance_matrix = gmx.geodesic_matrix(hexagons_gdf,
                                                     feedstock_points_gdf,
                                                     desc="Calculating distance to mines")
    
    demand_points_gdf["nearest hexidx"] = [gmx.find_nearest_hex(idx, hexagon_to_demand_distance_matrix) for idx in demand_points_gdf.index]
    feedstock_points_gdf["nearest hexidx"] = [gmx.find_nearest_hex(idx, hexagon_to_feedstock_distance_matrix) for idx in feedstock_points_gdf.index]
    
    #%% calculate road construction costs
    if road_construction:
        hexagons_road_construction = hexagons_gdf.apply(gmx.calculate_road_construction,
                                                                       args=[infra_data, country_parameters],
                                                                       axis=1)
    
    # may use to keep track of separate costs
    # road_construction_costs_to_demand = pd.DataFrame(np.empty((num_hex, num_dem)),
    #                                                  index=hexagons_gdf.index,
    #                                                  columns=demand_points_gdf.index)
    # road_construction_costs_to_generation = pd.DataFrame(np.empty((num_hex, num_gen)),
    #                                                  index=hexagons_gdf.index,
    #                                                  columns=feedstock_points_gdf.index)
        
   
    #%% meeting demand
    # =============================================================================
    # iterate through the list of demands. For each demand, calculate the
    # solution to meeting that demand from a Cu Concentrate feedstock (obtained
    # from the nearest mine) from each hexagon.
    # =============================================================================
    # iterate first over the demands
    for dix in tqdm(demand_points_gdf.index, desc="Demand"):
        # demand_location = Point(demand_center_list.loc[d,'Lat [deg]'],
        #                         demand_center_list.loc[d,'Lon [deg]'])
        demand = demand_points_gdf.loc[dix,:]
        demand_state = demand["Demand state"]
        demand_hix = demand_points_gdf["nearest hexidx"][dix]
        if demand_state not in demand_states:
            raise NotImplementedError(f'{demand_state} demand not supported.')
        hex_to_demand_dist = hexagon_to_demand_distance_matrix.loc[:,dix] # could access these directly to save memory
        # hex_to_gen_dist = hexagon_to_feedstock_distance_matrix.loc[:,demand.name] # could access these directly to save memory
        product_quantity = demand['Annual demand [kg/a]']
        feedstock_quantity = demand['Annual demand [kg/a]'] / conversion_parameters.loc["Efficiency (kg product / kg feedstock)", demand_state]
        if feedstock_quantity > feedstock_points_gdf["Annual capacity [kg/a]"].sum():
            raise NotImplementedError(f'Not enough feedstock demand to meet {dix} {demand_state}')
          
        
        
        road_construction_costs = np.empty(num_hex)
        trucking_state_to_demand = np.empty(num_hex, dtype='<U20')
        total_trucking_costs =  np.empty(num_hex)
        rail_construction_costs = np.empty(num_hex)
        train_state_to_demand = np.empty(num_hex, dtype='<U20')
        total_train_costs =  np.empty(num_hex)
        pv_capacities = np.empty(num_hex)
        wind_capacities = np.empty(num_hex)
        battery_capacities = np.empty(num_hex)
        lcoms = np.empty(num_hex)
        total_lcoms = np.empty(num_hex)
        
        # iterate over each hexagon
        # for hix in tqdm(hexagons_gdf.index, desc="Hexagon"):
        for h, hix in enumerate(hexagons_gdf.index):
            hexagon = hexagons_gdf.loc[hix, :]
                        
            # determine feedstock sources
            feedstock_sources, feedstock_ranked_idxs = gmx.determine_feedstock_sources(feedstock_points_gdf,
                                                                hexagon_to_feedstock_distance_matrix,
                                                                hix,
                                                                feedstock_quantity)
            
            # =============================================================================
            # cost of road transport from facility to demand
            # =============================================================================
            
            # cost of road construction
            demand_road_construction_cost = 0
            
            if road_construction:
                if hix==demand_hix: # demand is in same hexagon
                    # only need to account for 1 road construction
                    demand_road_construction = hexagons_road_construction[hix]
                else:
                    # build road for hexagon and demand hexagon
                    demand_road_construction = (hexagons_road_construction[hix] + 
                                                hexagons_road_construction[demand_hix])
            
            # calculate road transport to demand
            demand_trucking_cost_per_kg, demand_trucking_state = gmx.mineral_trucking_costs(demand_state,
                                                                                     hexagon_to_demand_distance_matrix.loc[hix, dix],
                                                                                     product_quantity,
                                                                                     country_parameters.loc[hexagons_gdf.loc[hix, 'country'], 'Infrastructure interest rate'],
                                                                                     )
            
            # =============================================================================
            # cost of road transport from feedstock to facility
            # =============================================================================
            
            # feedstock road construction
            feedstock_road_construction = 0
            if road_construction:
                for f in feedstock_sources.index:
                    feedstock_hix = feedstock_points_gdf["nearest hexidx"][f]
                    if hix==feedstock_hix: # feedstock is in same hexagon
                        # road construction already accounted for in demand
                        pass
                    else:
                        # build road for feedstock hexagon
                        feedstock_road_construction += hexagons_road_construction[feedstock_hix]
            
            # calculate road transport from feedstock
            feedstocks_trucking_cost = 0
            for f in feedstock_sources.index:
                feedstock_quantity = feedstock_sources.loc[f, "Feedstock used [kg/a]"]
                feedstock_trucking_cost_per_kg, demand_trucking_state = gmx.mineral_trucking_costs("CuConcentrate",
                                                                                         hexagon_to_demand_distance_matrix.loc[hix, dix],
                                                                                         feedstock_quantity,
                                                                                         country_parameters.loc[hexagons_gdf.loc[hix, 'country'], 'Infrastructure interest rate'],
                                                                                         )
                feedstocks_trucking_cost += feedstock_trucking_cost_per_kg * feedstock_quantity
            feedstocks_trucking_cost_per_kg_product = feedstocks_trucking_cost / product_quantity
            
            # =============================================================================
            # combined demand and feedstock road transport
            # =============================================================================
            road_construction_costs[h] = demand_road_construction_cost + feedstock_road_construction
            total_trucking_costs[h] = demand_trucking_cost_per_kg + feedstocks_trucking_cost_per_kg_product
            
            
            # =============================================================================
            # cost of rail transport from facility to demand
            # =============================================================================
            
            # cost of rail construction
            demand_rail_construction_cost = 0
            
            # =============================================================================
            # cost of rail transport from feedstock to facility
            # =============================================================================
            
            # feedstock rail construction
            feedstock_rail_construction = 0
            
            
            # =============================================================================
            # energy optimisation of facility
            # =============================================================================
            
            # trucking schedule
            demand_trucking_schedule = gmx.demand_schedule(product_quantity, demand_state)
            
            
            # facility energy optimisation
            if demand_state=='CuConcentrate':
                lcom=0
                wind_capacity=0
                solar_capacity=0
                battery_capacity=0
            else:
                (lcom,
                 wind_capacity,
                 solar_capacity,
                 battery_capacity) = gmx.optimize_facility(wind_profile.sel(hexagon = hix),
                                        pv_profile.sel(hexagon = hix),
                                        wind_profile.time,
                                        demand_trucking_schedule,
                                        demand_state,
                                        conversion_parameters.loc["Electricity demand (kWh per kg product)", demand_state],
                                        hexagons_gdf.loc[hix,'theo_turbines'],
                                        hexagons_gdf.loc[hix,'theo_pv'],
                                        country_parameters.loc[hexagons_gdf.loc[hix, "country"]],
                                        )
                    # if not enough renewables:
                    # set to nan or use grid
            pv_capacities[h] = solar_capacity
            wind_capacities[h] = wind_capacity
            battery_capacities[h] = battery_capacity
            lcoms[h] = lcom
            total_lcoms[h] = lcom + total_trucking_costs[h]
                
                
            # calculate cost of mine operation (ore to concentrate)
        hexagons_gdf[f'{dix} road construction costs'] = road_construction_costs/product_quantity
        hexagons_gdf[f'{dix} trucking transport costs'] = total_trucking_costs # cost of road construction, supply conversion, trucking transport, and demand conversion
        hexagons_gdf[f'{dix} PV capacity'] = pv_capacities
        hexagons_gdf[f'{dix} PV capacity'] = wind_capacities
        hexagons_gdf[f'{dix} PV capacity'] = battery_capacities
        hexagons_gdf[f'{dix} lcomf'] = lcoms
        hexagons_gdf[f'{dix} total lcom'] = total_lcoms
        
        # hexagons_gdf[f'{dix} trucking state'] = trucking_states # cost of road construction, supply conversion, trucking transport, and demand conversion
        # hexagons_gdf[f'{dix} pipeline transport and conversion costs'] = pipeline_costs # cost of supply conversion, pipeline transport, and demand conversion
    hexagons_gdf.to_file('Resources/hex_final.geojson', driver='GeoJSON', encoding='utf-8')
