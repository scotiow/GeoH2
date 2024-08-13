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
import numpy as np
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'


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
    
    
#%% standard plot
def pivot_choropleth(hex_gdf, boundary_gdf, mines_gdf, demand_gdf,
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
    filtered_df = hex_gdf.filter(like='total offgrid lcom', axis=1)
    hex_gdf['Tot'] = filtered_df.apply(lambda row: row.max() if not row.isna().all() else np.nan, axis=1)
    
    # Create a plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    # Plot the choropleth
    if continuous:
        hex_gdf.plot(column='Tot', alpha=0.7,
                            ax=ax, legend=True, legend_kwds={"label":'Levilised cost (euros/kg/year)'
                                                             })
    else:
        hex_gdf.plot(column='Tot', alpha=0.7,
                     scheme=method,
                            ax=ax, k=10, legend=True, legend_kwds={"loc": "center left",
                                                                   "bbox_to_anchor":(1,0.5)})

    # Plot the boundary layer
    boundary_gdf.boundary.plot(ax=ax, edgecolor='black')
    
    # # plot mine locations
    # mines_gdf.plot(ax=ax, marker='s', color='black', markersize=50, )
    
    # demand locations
    demand_gdf.plot(ax=ax, marker='o', color='red', markersize=10,)
    
    # Remove the axes
    ax.set_axis_off()
    
    
    # Show the plot
    plt.show()
    return hex_gdf

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
    
def grid_cost_breakdown(hex_gdf, demand_gdf):
    demand_idx = 0
    hex_idx = hex_gdf[hex_gdf["{} total offgrid lcom".format(demand_gdf.index[demand_idx])].notnull()].index[0]
    
    df_tot = hex_gdf[["{} total offgrid lcom".format(demand_gdf.index[demand_idx]),
                  "{} total grid lcom".format(demand_gdf.index[demand_idx]),
                  "{} total mix lcom".format(demand_gdf.index[demand_idx])]]
    
    df = hex_gdf[["{} offgrid lcomf (euros/kg/year)".format(demand_gdf.index[demand_idx]),
                  "{} annual facility costs (euros/kg/year)".format(demand_gdf.index[demand_idx]),
                  "{} Total road transport costs (euros/kg/year)".format(demand_gdf.index[demand_idx]),
                  "{} mix offgrid lcomf (euros/kg/year)".format(demand_gdf.index[demand_idx]),
                  "{} annual facility costs (euros/kg/year)".format(demand_gdf.index[demand_idx]),
                  "{} Total road transport costs (euros/kg/year)".format(demand_gdf.index[demand_idx]),
                  "{} grid lcomf (euros/kg/year)".format(demand_gdf.index[demand_idx]),
                  "{} annual facility costs (euros/kg/year)".format(demand_gdf.index[demand_idx]),
                  "{} Total road transport costs (euros/kg/year)".format(demand_gdf.index[demand_idx]),
                                              
                  ]]
    
    df = df.loc[hex_idx,:]
    fig, ax = plt.subplots()
    ax.bar(0, df["{} offgrid lcomf (euros/kg/year)".format(demand_gdf.index[demand_idx])],
           label='Energy',
           color='r')
    ax.bar(0, df["{} annual facility costs (euros/kg/year)".format(demand_gdf.index[demand_idx])],
           bottom=df["{} offgrid lcomf (euros/kg/year)".format(demand_gdf.index[demand_idx])],
           label='Facility', color='b')
    ax.bar(0, df["{} Total road transport costs (euros/kg/year)".format(demand_gdf.index[demand_idx])],
           bottom=df["{} annual facility costs (euros/kg/year)".format(demand_gdf.index[demand_idx])],
           label='Transport', color='g')
    
    
    ax.bar(1, df["{} grid lcomf (euros/kg/year)".format(demand_gdf.index[demand_idx])],
           color='r')
    ax.bar(1, df["{} annual facility costs (euros/kg/year)".format(demand_gdf.index[demand_idx])],
           bottom=df["{} grid lcomf (euros/kg/year)".format(demand_gdf.index[demand_idx])], 
           color='b')
    ax.bar(1, df["{} Total road transport costs (euros/kg/year)".format(demand_gdf.index[demand_idx])],
           bottom=df["{} annual facility costs (euros/kg/year)".format(demand_gdf.index[demand_idx])],
           color='g')
        
    
    
    # ax.bar(2, df["{} total mix lcom".format(demand_gdf.index[demand_idx])], label='Mix grid')
    
    ax.legend()
    plt.show()
    
def cost_breakdown(hex_gdf, demand_gdf):
    
    df = pd.DataFrame(columns=["demand", "cost", "component", "type"])
    for typ in ["offgrid", "grid"]:
        for demand in demand_gdf.index:
            hex_idx = hex_gdf[hex_gdf["{} total {} lcom".format(demand, typ)].notnull()].index[0]
            for component, comp_label in zip(["Energy",
                                              "Feedstock transport",
                                              "Product transport",
                                              "Facility"],
                                             ["{} {} lcomf (euros/kg/year)".format(demand, typ),
                                              "{} feedstock trucking transport costs (euros/kg/year)".format(demand),
                                              "{} product trucking transport costs (euros/kg/year)".format(demand),
                                              "{} annual facility costs (euros/kg/year)".format(demand)
                                              ]):
                df = pd.concat([df, pd.DataFrame({
                    "demand": [demand],
                    "cost": [hex_gdf.loc[hex_idx, comp_label]],
                    "component": [component],
                    "type": [typ]})], ignore_index=True)
            
            other_cost = (hex_gdf.loc[hex_idx, "{} total offgrid lcom".format(demand)]
                          - df.loc[(df["demand"]==demand) & (df["type"]=="offgrid") ,"cost"].sum())
            df = pd.concat([df, pd.DataFrame({
                "demand": [demand],
                "cost": [other_cost],
                "component": ["other"],
                "type": ["offgrid"]})], ignore_index=True)
        
    fig = px.bar(df, x="demand", y="cost", color="component", title="test", facet_row="type")
    fig.show()
    return df

def sub_df(hex_gdf, demand):
    
    df = hex_gdf.filter(like=demand, axis=1)
    return df.dropna(how='all').T


if __name__=="__main__":
    
    hex_gdf, boundary_gdf, demand_gdf, mines_gdf = import_data()
    standard_choropleth(hex_gdf, boundary_gdf, mines_gdf, demand_gdf,
                            method="equalinterval", continuous=True)
    
    hex_piv = pivot_choropleth(hex_gdf, boundary_gdf, mines_gdf, demand_gdf,
                            method="equalinterval", continuous=True)
    grid_cost_breakdown(hex_gdf, demand_gdf)
    
    test = cost_breakdown(hex_gdf, demand_gdf)
    
    cambishi = sub_df(hex_gdf, demand_gdf.index[0])
    mulfulira = sub_df(hex_gdf, demand_gdf.index[1])
    nichanga = sub_df(hex_gdf, demand_gdf.index[2])
