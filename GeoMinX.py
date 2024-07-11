# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:15:32 2024

@author: Scot Wheeler
"""

import json
import geopandas as gpd
import pandas as pd
import GeoMinX_functions as gmx
import numpy as np
import logging
from tqdm import tqdm
import os
from shapely.geometry import Point
import warnings
from shapely.errors import ShapelyDeprecationWarning
# this ignores the following warning when using atlite: 
    # ShapelyDeprecationWarning: STRtree will be changed in 2.0.0 and will not be compatible with versions < 2.
# be aware, this may lead to a future error if using newer versions of Shapely
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
logging.basicConfig(level=logging.ERROR)



# =============================================================================
# Code Configuration Parameters
#   should these be added to the global_data parameter file?
# =============================================================================
country_names = ["Zambia"]  # Multiple countries is not yet implemented, see note below.
demand_states = ['CuAnode','CuCathode']
combine_spider_glaes = False


# =============================================================================
# Start of programme
# =============================================================================

# Create Resources folder to save results if it doesn't already exist
if not os.path.exists('Resources'):
    os.makedirs('Resources')

# =============================================================================
# NOTE:
#    Multiple countries is not yet implemented. The high level country loop
#    exits here purely as a place holder for future development.
# =============================================================================

# combine spider and glaes inputs. This replaces the need for GeoH2_data_prep.
if combine_spider_glaes:
    gmx.combine_glaes_spider(country_names)

for country_name in country_names:
    country_name_clean = gmx.clean_country_name(country_name)
    print(country_name)
    
    # load parameter files
    (infra_data, global_data, mineral_data, demand_center_list,
            generation_center_list, country_parameters) = gmx.import_parameter_files()
    conversion_parameters = gmx.import_conversion_parameters(demand_states) # from feedstock
    
    road_construction = global_data['Road construction allowed']
    rail_construction = global_data['Rail construction allowed']
    
    #%% Import hexagon inputs
    
    _, hexagons_gdf = gmx.load_hexagons(country_name_clean, country_parameters) # import only those associated with the country defined above
    num_hex = hexagons_gdf.shape[0]
    
    #%% convert demand and feedstock locations to geodataframes
    demand_points_gdf = gpd.GeoDataFrame(demand_center_list, geometry=[Point(xy) for xy in zip(demand_center_list['Lon [deg]'], demand_center_list['Lat [deg]'])]).set_crs(epsg=4326)
    num_dem = demand_points_gdf.shape[0]
    
    feedstock_points_gdf = gpd.GeoDataFrame(generation_center_list, geometry=[Point(xy) for xy in zip(generation_center_list['Lon [deg]'], generation_center_list['Lat [deg]'])]).set_crs(epsg=4326)
    num_gen = feedstock_points_gdf.shape[0]
    
    #%% import weather data, pv and wind profiles
    pv_profile, wind_profile = gmx.get_pv_wind_profiles(hexagons_gdf)
    
    #%% calculate distances
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
    # for dix in tqdm(demand_points_gdf.index, desc="Demand"):
    demand_sites = demand_points_gdf.shape[0]
    for d, dix in enumerate(demand_points_gdf.index):
        if dix!="Livingstone":
            continue
    
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
          
        # facility costs
        annual_facility_costs = np.empty(num_hex)
        
        # road costs
        road_construction_costs = np.empty(num_hex)  # total (product + feedstock) cost
        trucking_state_to_demand = np.empty(num_hex, dtype='<U20')  
        total_trucking_costs =  np.empty(num_hex)  # total (product + feedstock) trucking cost per kg product
        demand_trucking_costs =  np.empty(num_hex)
        feedstock_trucking_costs =  np.empty(num_hex)
        
        
        # rail costs
        rail_construction_costs = np.empty(num_hex)
        train_state_to_demand = np.empty(num_hex, dtype='<U20')
        total_train_costs =  np.empty(num_hex)
        
        # off-grid costs
        pv_capacities = np.empty(num_hex)
        wind_capacities = np.empty(num_hex)
        battery_capacities = np.empty(num_hex)
        offgrid_lcoms = np.empty(num_hex)
        
        # grid costs
        grid_lcoms = np.empty(num_hex)
        
        # total costs
        total_offgrid_lcoms = np.empty(num_hex)
        total_grid_lcoms = np.empty(num_hex)
        
        # micellaneous
        feedstock_locs = []
        
        # iterate over each hexagon
        for h, hix in enumerate(tqdm(hexagons_gdf.index, desc=f'Demand {d+1}/{demand_sites}')):
        # for h, hix in enumerate(hexagons_gdf.index):
            hexagon = hexagons_gdf.loc[hix, :]
                        
            # determine feedstock sources
            feedstock_sources, feedstock_ranked_idxs = gmx.determine_feedstock_sources(feedstock_points_gdf,
                                                                hexagon_to_feedstock_distance_matrix,
                                                                hix,
                                                                feedstock_quantity)
            feedstock_locs.append(json.dumps(list(feedstock_ranked_idxs.values)))
            
            # =============================================================================
            # cost of road transport from facility to demand
            # =============================================================================
            
            # cost of road construction (total)
            demand_road_construction = 0
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
                source_quantity = feedstock_sources.loc[f, "Feedstock used [kg/a]"]
                feedstock_trucking_cost_per_kg, demand_trucking_state = gmx.mineral_trucking_costs("CuConcentrate",
                                                                                         hexagon_to_feedstock_distance_matrix.loc[hix, f],
                                                                                         source_quantity,
                                                                                         country_parameters.loc[hexagons_gdf.loc[hix, 'country'], 'Infrastructure interest rate'],
                                                                                         )
                feedstocks_trucking_cost += feedstock_trucking_cost_per_kg * source_quantity
            feedstocks_trucking_cost_per_kg_product = feedstocks_trucking_cost / product_quantity
                    
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
            
            
            # trucking schedule
            demand_trucking_schedule = gmx.demand_schedule_constant(product_quantity, demand_state)
            
            # =============================================================================
            # cost of processing facility
            # =============================================================================
            
            facility_annual_cost = gmx.mineral_conversion_stand(demand_state,
                                                                product_quantity,
                                                                country_parameters.loc[hexagons_gdf.loc[hix, 'country'],'Plant interest rate'])
            facility_annual_cost_per_kg = facility_annual_cost / product_quantity
            
            # =============================================================================
            # energy optimisation of facility
            # =============================================================================
            
            # grid construction?
            
            
            # grid power
            # grid_energy_cost = (conversion_parameters.loc["Electricity demand (kWh per kg product)", demand_state]
            #                     * demand_trucking_schedule
            #                     * country_parameters.loc[hexagons_gdf.loc[hix, "country"], "Electricity price (euros/kWh)"]).sum()[0]
            # grid_energy_cost_per_kg = grid_energy_cost / product_quantity
            # print(grid_energy_cost_per_kg)
            
            (grid_energy_cost_per_kg, _, _, _,
             grid_capacity,
             network) = gmx.optimize_ES_gridconnected(wind_profile.sel(hexagon = hix),
                                    pv_profile.sel(hexagon = hix),
                                    wind_profile.time,
                                    demand_trucking_schedule,
                                    demand_state,
                                    conversion_parameters.loc["Electricity demand (kWh per kg product)", demand_state],
                                    0,
                                    0,
                                    country_parameters.loc[hexagons_gdf.loc[hix, "country"]],
                                    battery_p_max=0)
            
            
            # =============================================================================
            # energy optimisation of facility
            # =============================================================================
            
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
                 battery_capacity,
                 network) = gmx.optimize_offgrid_facility(wind_profile.sel(hexagon = hix),
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
            
            
            
            
            # =============================================================================
            # energy optimisation of facility
            # =============================================================================
            
            heat_unit_demand = conversion_parameters.loc["Heat demand (kWh per kg product)", demand_state]
            heat_demand = heat_unit_demand * product_quantity
            
            
            diesel_unit_demand = conversion_parameters.loc["Diesel demand (kWh per kg product)", demand_state]
            diesel_demand = diesel_unit_demand * product_quantity
            diesel_cost = diesel_demand * country_parameters.loc[hexagons_gdf.loc[hix, "country"], "Diesel price (euros/kWh)"]
            
            # =============================================================================
            # update output vectors
            # =============================================================================
            
            # facility costs
            annual_facility_costs[h] = facility_annual_cost_per_kg
           
            # road costs
            road_construction_costs[h] = (demand_road_construction + feedstock_road_construction) / product_quantity
            demand_trucking_costs[h] = demand_trucking_cost_per_kg
            feedstock_trucking_costs[h] = feedstocks_trucking_cost_per_kg_product
            total_trucking_costs[h] = demand_trucking_cost_per_kg + feedstocks_trucking_cost_per_kg_product
            
            # rail costs
            rail_construction_costs[h] = 0
            total_train_costs[h] =  0
            
            # off-grid costs
            pv_capacities[h] = solar_capacity
            wind_capacities[h] = wind_capacity
            battery_capacities[h] = battery_capacity
            offgrid_lcoms[h] = lcom
            
            # grid costs
            grid_lcoms[h] = grid_energy_cost_per_kg
            
            # total costs
            if lcom==np.nan:
                total_offgrid_lcoms[h] = np.nan
            else:
                total_offgrid_lcoms[h] = (lcom
                                  + facility_annual_cost_per_kg
                                  + road_construction_costs[h] 
                                  + total_trucking_costs[h])
            
            total_grid_lcoms[h] = (grid_energy_cost_per_kg
                              + facility_annual_cost_per_kg
                              + road_construction_costs[h] 
                              + total_trucking_costs[h])
            # calculate cost of mine operation (ore to concentrate)
        
        # =============================================================================
        # update demand outputs
        # =============================================================================
        # facility costs
        hexagons_gdf[f'{dix} annual facility costs (euros/a/kg)'] = facility_annual_cost_per_kg 
        # road costs
        hexagons_gdf[f'{dix} road construction costs (euros/a/kg)'] = road_construction_costs
        hexagons_gdf[f'{dix} trucking transport costs (euros/a/kg)'] = total_trucking_costs # cost of road construction, supply conversion, trucking transport, and demand conversion
        hexagons_gdf[f'{dix} feedstock trucking transport costs (euros/a/kg)'] = feedstock_trucking_costs # cost of road construction, supply conversion, trucking transport, and demand conversion
        hexagons_gdf[f'{dix} product trucking transport costs (euros/a/kg)'] = demand_trucking_costs # cost of road construction, supply conversion, trucking transport, and demand conversion
        
        # rail costs
        # off-grid costs
        hexagons_gdf[f'{dix} PV capacity'] = pv_capacities
        hexagons_gdf[f'{dix} Wind capacity'] = wind_capacities
        hexagons_gdf[f'{dix} Battery capacity'] = battery_capacities
        hexagons_gdf[f'{dix} offgrid lcomf (euros/a/kg)'] = offgrid_lcoms
        # grid costs
        hexagons_gdf[f'{dix} grid lcomf (euros/a/kg)'] = grid_lcoms
        # total costs
        hexagons_gdf[f'{dix} total offgrid lcom'] = total_offgrid_lcoms
        hexagons_gdf[f'{dix} total grid lcom'] = total_grid_lcoms
        hexagons_gdf[f'{dix} feedstock locs'] = feedstock_locs
        
        # hexagons_gdf[f'{dix} trucking state'] = trucking_states # cost of road construction, supply conversion, trucking transport, and demand conversion
        # hexagons_gdf[f'{dix} pipeline transport and conversion costs'] = pipeline_costs # cost of supply conversion, pipeline transport, and demand conversion
    output_path = os.path.join("Data", "Resources", f"{country_name_clean}_hex_GeoX_final.geojson")
    hexagons_gdf.to_file(output_path, driver='GeoJSON', encoding='utf-8')
