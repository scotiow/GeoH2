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
import mapclassify as mc

#%% import data
final_hex_data = gpd.read_file('Resources/hex_final.geojson')

zambia_outline = gpd.read_file("Resources/Zambia.gpkg")
final_hex_crop = final_hex_data[final_hex_data.within(zambia_outline.unary_union)]

demand_center_list = pd.read_excel("Parameters/demand_parameters.xlsx",
                                   sheet_name='Demand centers',
                                   index_col='Demand center',
                                   )
generation_center_list = pd.read_excel("Parameters/demand_parameters.xlsx",
                                   sheet_name='Generation centers',
                                   index_col='Generation center',
                                   )
demand_gdf = gpd.GeoDataFrame(demand_center_list, geometry=[Point(xy) for xy in zip(demand_center_list['Lon [deg]'], demand_center_list['Lat [deg]'])]).set_crs(epsg=4326)
mines_gdf = gpd.GeoDataFrame(generation_center_list, geometry=[Point(xy) for xy in zip(generation_center_list['Lon [deg]'], generation_center_list['Lat [deg]'])]).set_crs(epsg=4326)

#%% main plot
# Create a plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Plot the choropleth
# Create an equal count (quantiles) classification scheme
quantiles = mc.Quantiles(final_hex_data['Livingstone total offgrid lcom'], k=10)
final_hex_data.assign(cl=quantiles.yb).plot(column='cl', alpha=0.7,
                    ax=ax, legend=True,
                    legend_kwds={"label": "Levilized cost of Cu product (euros/kg)"})

# Plot the boundary layer
zambia_outline.boundary.plot(ax=ax, edgecolor='black')

# plot mine locations
mines_gdf.plot(ax=ax, marker='s', color='black', markersize=50, label="Cu Mines")

# demand locations
demand_gdf.plot(ax=ax, marker='o', color='red', markersize=50, label="Cu Demand")

# add basemap
# cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)

# Remove the axes
ax.set_axis_off()

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='lower left')

# Show the plot
plt.show()

#%% offgrid lcom
# Create a plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Plot the choropleth
final_hex_data.plot(column='Livingstone offgrid lcomf (euros/a/kg)', alpha=0.7,
                    ax=ax, legend=True,
                    legend_kwds={"label": "Levilized cost of Cu product (euros/kg)"})

# Plot the boundary layer
zambia_outline.boundary.plot(ax=ax, edgecolor='black')

# plot mine locations
mines_gdf.plot(ax=ax, marker='s', color='black', markersize=50, label="Cu Mines")

# demand locations
demand_gdf.plot(ax=ax, marker='o', color='red', markersize=50, label="Cu Demand")

# Remove the axes
ax.set_axis_off()

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='lower left')

# Show the plot
plt.show()

#%% mines and demand
# Create a plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)


# Plot the boundary layer
zambia_outline.boundary.plot(ax=ax, edgecolor='black')

# plot mine locations
mines_gdf.plot(ax=ax, marker='s', color='black', markersize=50, label="Cu Mines")

# demand locations
demand_gdf.plot(ax=ax, marker='o', color='red', markersize=50, label="Cu Demand")

# add basemap
cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, crs=zambia_outline.crs)

# Remove the axes
ax.set_axis_off()

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='lower left')

# Show the plot
plt.show()

#%% offgrid lcom
# Create a plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)


# Plot the boundary layer
zambia_outline.boundary.plot(ax=ax, edgecolor='black')

# Plot the choropleth
final_hex_crop.plot(column='theo_pv', alpha=0.7,
                    ax=ax, legend=True,
                    legend_kwds={"label": "Wind potential"})

# add basemap
cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, crs=zambia_outline.crs)

# Remove the axes
ax.set_axis_off()

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='lower left')

# Show the plot
plt.show()

