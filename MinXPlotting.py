# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 08:45:28 2024

@author: Scot Wheeler
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
from shapely import Point
import pandas as pd
from mapclassify import classify
import os
import matplotlib as mpl


#%% import data
def import_data(country_name="Zambia"):
    result_file_path = os.path.join("Resources", f"{country_name}_hex_GeoX_final.geojson")
    outline_file_path = os.path.join("Resources", f"{country_name}.gpkg")
    demand_gen_file_path = os.path.join("Parameters", "demand_parameters.xlsx")
    
    final_hex_data = gpd.read_file(result_file_path)
    
    outline = gpd.read_file(outline_file_path)
    # final_hex_crop = final_hex_data[final_hex_data.within(zambia_outline.unary_all)]
    
    demand_center_list = pd.read_excel(demand_gen_file_path,
                                       sheet_name='Demand centers',
                                       index_col='Demand center',
                                       )
    generation_center_list = pd.read_excel(demand_gen_file_path,
                                       sheet_name='Generation centers',
                                       index_col='Generation center',
                                       )
    demand_gdf = gpd.GeoDataFrame(demand_center_list, geometry=[Point(xy) for xy in zip(demand_center_list['Lon [deg]'], demand_center_list['Lat [deg]'])]).set_crs(epsg=4326)
    mines_gdf = gpd.GeoDataFrame(generation_center_list, geometry=[Point(xy) for xy in zip(generation_center_list['Lon [deg]'], generation_center_list['Lat [deg]'])]).set_crs(epsg=4326)

    return final_hex_data, outline, demand_gdf, mines_gdf
#%% standard plot
def standard_choropleth(hex_gdf, boundary_gdf, mines_gdf, demand_gdf,
                        column=None, label=None, method="EqualInterval", continuous=True):
    """
    See https://geographicdata.science/book/notebooks/05_choropleth.html#equal-intervals for choropleth tips and methods detail

    Parameters
    ----------
    hex_gdf : TYPE
        DESCRIPTION.
    boundary_gdf : TYPE
        DESCRIPTION.
    mines_gdf : TYPE
        DESCRIPTION.
    demand_gdf : TYPE
        DESCRIPTION.
    column : TYPE, optional
        DESCRIPTION. The default is None.
    label : TYPE, optional
        DESCRIPTION. The default is None.
    method : TYPE, optional
        DESCRIPTION. The default is "EqualInterval".

    Returns
    -------
    None.

    """
    if isinstance(column, type(None)):
        column = "{} total offgrid lcom".format(demand_gdf.index[0])
    if isinstance(label, type(None)):
        label=column
    # Create a plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot the choropleth
    if continuous:
        hex_gdf.plot(column=column, alpha=0.7,
                            ax=ax, legend=True, legend_kwds={"label":label,
                                                             })
    else:
        hex_gdf.plot(column=column, alpha=0.7,
                     scheme=method,
                            ax=ax, k=10, legend=True, legend_kwds={"loc": "center left",
                                                                   "bbox_to_anchor":(1,0.5)})
    
    # Plot the boundary layer
    boundary_gdf.boundary.plot(ax=ax, edgecolor='black')
    
    # plot mine locations
    mines_gdf.plot(ax=ax, marker='s', color='black', markersize=50, )
    
    # demand locations
    demand_gdf.plot(ax=ax, marker='o', color='red', markersize=50,)
    
    # Remove the axes
    ax.set_axis_off()
    
    
    # Show the plot
    plt.show()
    
#%% mines and demand
def mines_and_demand(boundary_gdf, mines_gdf, demand_gdf):
    # Create a plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    
    
    # Plot the boundary layer
    boundary_gdf.boundary.plot(ax=ax, edgecolor='black')
    
    # plot mine locations
    mines_gdf.plot(ax=ax, marker='s', color='black', markersize=50, label="Mines")
    
    # demand locations
    demand_gdf.plot(ax=ax, marker='o', color='red', markersize=50, label="Demand")
    
    # add basemap
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, crs=boundary_gdf.crs)
    
    # Remove the axes
    ax.set_axis_off()
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower left')
    
    # Show the plot
    plt.show()
    



if __name__=="__main__":
    
    hex_gdf, boundary_gdf, demand_gdf, mines_gdf = import_data()
    standard_choropleth(hex_gdf, boundary_gdf, mines_gdf, demand_gdf,
                            method="equalinterval", continuous=True)
    
    