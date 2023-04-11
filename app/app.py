# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import os
import dash
from dash import Dash, dcc, html, ctx
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import dash_daq as daq
import plotly.express as px
import pandas as pd

import time
import datetime
from dash import DiskcacheManager, CeleryManager, Input, Output, html
import shutil
import numpy as np
import json 
import sys

import localnmf
from localnmf.superpixel_analysis_ring import superpixel_init
import scipy
import scipy.sparse
import torch_sparse
import localnmf 
from localnmf import superpixel_analysis_ring
import os
import numpy as np
import scipy
import scipy.sparse
import torch_sparse
import torch
import localnmf 
from localnmf import superpixel_analysis_ring
from localnmf.pmd_video import PMDVideo
from localnmf.pmd_video import local_correlation_mat
from localnmf_functions import get_single_pixel_corr_img
import math
import jax

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".70"

import tifffile

### PRODUCTION VS NON PRODUCTION WORKER MANAGEMENT 
# if 'REDIS_URL' in os.environ:
#     # Use Redis & Celery if REDIS_URL set as an env variable
#     from celery import Celery
#     celery_app = Celery(__name__, broker=os.environ['REDIS_URL'], backend=os.environ['REDIS_URL'])
#     background_callback_manager = CeleryManager(celery_app)

# else:
# Diskcache for non-production apps when developing locally
import diskcache

large_cache_size = 1073741824 * 15
cache = diskcache.Cache("./cache", size_limit = large_cache_size )

print("THE DEFAULT SIZE OF CACHE IS {}".format(cache.size_limit))
background_callback_manager = DiskcacheManager(cache)


def list_dir(path):
    print("input to list_dir is {}".format(path))
    return os.listdir(path) + [".."]

# load_figure_template('LUX')
stylesheets = [dbc.themes.LUX]
app = Dash(__name__, background_callback_manager=background_callback_manager, external_stylesheets=stylesheets)


##Globally used data structures/info
'''
Structure of this app: we define the data structures used in the algorithm (for e.g. the dictionary containing all parameters)
as global variables, which can be modified by the callback functions
'''

cache['no_file_flag'] = ""
cache['navigated_folder'] = os.getcwd()
cache['navigated_file'] = cache['no_file_flag']

cache['PMD_flag'] = False #Indicates whether PMD has been run or not
cache['demix_flag'] = False #Indicates whether demixing has been run or not 
cache['PMD_object'] = None

cache['register_run_flag'] = True #This flag will be modified if instead do not want to register the data


mc_params = {
    'register':True,
    'dx':2,
    'dy':2,
    'devel':True,
    'max_shift_in_um':[50,50],
    'max_deviation_rigid':5,
    'patch_motion_um':[17,17],
    'overlaps':[24,24],
    'niter_rig':2,
    'niter_els':2,
    'pw_rigid':True,
    'use_gSig_filt':False,
    'gSig_filt':[3,3],
}

pmd_params = {
    'block_height':20,
    'block_width':20,
    'overlaps_height':10,
    'overlaps_width':10,
    'window_length':6000,
    'background_rank':15,
    'deconvolve':True,
}

localnmf_params = {
        'init': ['custom', 'lnmf', 'lnmf', 'lnmf'],
        'num_passes':2,
        'superpixels_corr_thr':[0.9, 0.85, 0.9, 0.86],
        'length_cut':[3,5,2,2],
        'th':[2,2,2,2],
        'pseudo_2':[0.1, 0.1, 0.1, 0.1],
        'corr_th_fix':0.55,
        'switch_point':5,
        'corr_th_fix_sec':0.7,
        'corr_th_del':0.2,
        'max_allow_neuron_size':0.15,
        'merge_corr_thr':0.7,
        'merge_overlap_thr':0.7,
        'r':20,
        'residual_cut':[0.5, 0.6, 0.6, 0.6],
        'num_plane': 1,
        'patch_size': [100,100],
        'maxiter': 25,
        'update_after':4, 
        'plot_en': False,
        'skips':0,
        'text': True,
        'sb': True,
}



cache['mc_params'] = mc_params
cache['pmd_params'] = pmd_params
cache['localnmf_params'] = localnmf_params
cache['demixing_results'] = None
cache['first_pass_init_method'] = 'masknmf'


img = np.random.rand(3,50,50)*0
mc_pmd_vis_frames = [i for i in range(100)]
fig_mc_pmd_plots = px.imshow(img, facet_col=0)
fig_mc_pmd_plots.update(layout_coloraxis_showscale=True)

img_name_list = ["No Results Yet", "No Results Yet", "No Results Yet"]
for i, name in enumerate(img_name_list):
    fig_mc_pmd_plots.layout.annotations[i]['text'] = name
    
   
###########
## MC + PMD dash components
###########

block_pmd_1 = html.Div(
    [
        html.P("Pick height block dimension between 10 and 50",  style={'textAlign': 'center'}),
        dbc.Input(type="number", id="block_pmd_1", min=10, max=50, value = pmd_params['block_height'], step=1,  style={'textAlign': 'center'}),
    ],
)

block_pmd_2 = html.Div(
    [
        html.P("Pick width block dimension between 10 and 50",  style={'textAlign': 'center'}),
        dbc.Input(type="number",id="block_pmd_2", min=10, max=50, value=pmd_params['block_width'], step=1, style={'textAlign': 'center'}),
    ],
)

block_pmd_placeholder_1 = html.Div(id='block_pmd_placeholder_1', children="")
block_pmd_placeholder_2 = html.Div(id='block_pmd_placeholder_2', children="")
    
    
    

#######
##This is for the pixel display
#######
trace = np.zeros((200))
indices = [i for i in range(1, trace.shape[0]+1)]
trace = pd.DataFrame(trace, columns = ['X'], index = indices)
fig_trace_vis = px.line(trace, y="X", 
                       labels={
                     "index": "Frame Number",
                     "X": "A.U.",
                 },)
fig_trace_vis.update_layout(title_text="After running registration + PMD, click pixels in above images to see PMD traces here", title_x=0.5)


#####
## This is for the first pass initialization display
#####
default_figsize = (40, 40)
cache['default_figsize'] = default_figsize
pixel_plot = np.zeros(default_figsize)
fig_local_corr = px.imshow(pixel_plot)
fig_local_corr.update_layout(title_text="Noise Variance Image: No Results Yet", title_x=0.5)
fig_local_corr.update_coloraxes(showscale=False)
fig_local_corr.update(layout_coloraxis_showscale=False)

pixel_plot = np.zeros(default_figsize)
fig_pixel_corr = px.imshow(pixel_plot)
fig_pixel_corr.update_layout(title_text="Pixelwise Corr. Image: No Results Yet", title_x=0.5)
fig_pixel_corr.update_coloraxes(showscale=False)
fig_pixel_corr.update(layout_coloraxis_showscale=False)
fig_pixel_corr.update_yaxes(showticklabels=False)




pixel_plot = np.zeros(default_figsize)
fig_superpixel = px.imshow(pixel_plot)
fig_superpixel.update_layout(title_text="Initialization Image: No Results Yet", title_x=0.5)
fig_superpixel.update_yaxes(showticklabels=False)



#####
## This is for the second pass superpixel display 
#####


pixel_plot = np.zeros(default_figsize)
fig_local_corr_secondpass = px.imshow(pixel_plot)
fig_local_corr_secondpass.update_layout(title_text="Residual Corr. Image: No Results Yet", title_x=0.5)
fig_local_corr_secondpass.update_coloraxes(showscale=False)
fig_local_corr_secondpass.update(layout_coloraxis_showscale=False)

pixel_plot = np.zeros(default_figsize)
fig_pixel_corr_secondpass = px.imshow(pixel_plot)
fig_pixel_corr_secondpass.update_layout(title_text="Pixelwise Corr. Image: No Results Yet", title_x=0.5)
fig_pixel_corr_secondpass.update_coloraxes(showscale=False)
fig_pixel_corr_secondpass.update(layout_coloraxis_showscale=False)
fig_pixel_corr_secondpass.update_yaxes(showticklabels=False)





pixel_plot = np.zeros(default_figsize)
fig_superpixel_secondpass = px.imshow(pixel_plot)
fig_superpixel_secondpass.update_layout(title_text="Superpixel Image: No Results Yet", title_x=0.5)
fig_superpixel_secondpass.update_yaxes(showticklabels=False)

######
## This is for visualizing the demixing outputs
######

pixel_plot = np.zeros(default_figsize)
fig_post_demixing_summary_image = px.imshow(pixel_plot)
fig_post_demixing_summary_image.update_layout(title_text="Source Extraction: No Results Yet", title_x=0.5)
fig_post_demixing_summary_image.update_coloraxes(showscale=False)
fig_post_demixing_summary_image.update(layout_coloraxis_showscale=False)

trace = np.zeros((200))
indices = [i for i in range(1, trace.shape[0]+1)]
trace = pd.DataFrame(trace, columns = ['X'], index = indices)
fig_post_demixing_pixewise_traces = px.line(trace, y="X", 
                       labels={
                     "index": "Frame Number",
                     "X": "A.U.",
                 },)
fig_post_demixing_pixewise_traces.update_layout(title_text="Pixel-wise demixing: No Results Yet", title_x=0.5)



### End of globally used data structures

controls = dcc.Dropdown(
        id="dropdown",
        options=[{"label": x, "value": x} for x in list_dir(cache['navigated_folder'])],
        value="",
        style={'width':'100%', 'margin-top':'20px'},
    )




SIDEBAR_STYLE = {
    # "position": "fixed",
    # "top": 0,
    # "left": 0,
    # "bottom": 0,
    # "width": "24rem",
    # "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "height":"90vh"
}



sidebar = html.Div(
    [
        html.H2("Preprocessing",  style={'textAlign': 'center'}),
        html.Hr(),
        html.H2(
            "Step 1: Select File",  style={'textAlign': 'center'}
        ),
        controls,\
        html.Br(),\
        html.Br(),\
        html.H4("File Selected: None",id="folder-files",  style={'textAlign': 'center'}),\
        html.H4("Folder: {}".format(cache['navigated_folder']), id="curr_folder",  style={'textAlign': 'center'}),\
        html.Br(),\
        html.Br(),\
        html.Br(),\
        html.Br(),\
        html.Br(id='boolean-switch-dumboutput'),\
        html.Hr(),\
        html.H2("Step 2: Register, Compress, Denoise Data ", style={'textAlign': 'center'}),\
        dbc.Row(
            [
                dbc.Col(
                    block_pmd_1,
                    width=6,
                ),\
                dbc.Col(
                    block_pmd_2,
                    width=6,
                ),\
            ]
        ),\
        dbc.Row(
            [
                dbc.Col(
                    block_pmd_placeholder_1,
                    width=6,
                ),\
                dbc.Col(
                    block_pmd_placeholder_2,
                    width=6,
                ),\
            ]
        ),\
        dbc.Row(
            [
                dbc.Col(
                    daq.BooleanSwitch(
                      on=True,
                      label="Register Data",
                      labelPosition="top",
                      id='register_flag',
                    ),\
                    # width = 3
                ),\
                dbc.Col(
                    dbc.Button(id="button_id", children="Run Job!"),\
                    # width = 9
                ),\
            ],\
            justify="evenly",\
        ),\
        
        html.Div(
                [
                    html.Div(id='placeholder', children=""),
                ]
        ),\
        
    ],
    style=SIDEBAR_STYLE,
)


sidebar_demixing = html.Div(
    [
        html.H2("Step 3: Demixing. Initialize neural signals and hit RUN",  style={'textAlign': 'center'}),\
        html.Div(
                    [
                        html.Div(id='placeholder_demix', children=""),
                    ]
        ),\
        dbc.Row(
            [
             dbc.Col(
                [
                     html.H5('''Select an initialization method. By default, the dense init will be run. To run the superpixels init instead, turn off the dense init and specify a superpixel threshold.''', style={'textAlign':'center'}),\
                ],\
                 width = 12
             ),\
            ]
        ),\
        dbc.Row(
           [
               dbc.Col(
                   daq.BooleanSwitch(
                      on=True,
                      label="Dense Data",
                      labelPosition="top",
                      id='masknmf_flag',
                    ),\
                   width=3
               ),\
               
               dbc.Col(
                [ 
                   dash.dcc.Slider(id='superpixel_slider',min=0.00,max=0.999,marks={0:'0', 0.1:'0.1', 0.2:'0.2', 0.3:'0.3', 0.4:'0.4', 0.5:'0.5', 0.6:'0.6',0.7:'0.7', 0.8:'0.8', 0.9:'0.9', 1:'1'},updatemode='mouseup',step=0.01,\
                                     value=0.0, disabled=True),\
                    html.H5("Superpixel Threshold", id="Slider Label", style={'textAlign': 'center'}),\
                ],\
                   width=9
               ),\
           ]
        ),\
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Button(id="button_id_demix", children="Run Demixing", style={'verticalalign':'center'}),\
                    ],\
                    width={"size": 6, "offset": 4},
                ),\
            ],\
        ),\
        
        html.Br(),\
        html.Br(),\
        html.Hr(),\
        
        html.H2("Step 4: (Optional) Multi-pass demixing for identifying remaining signals. Initialize signals and hit RUN",  style={'textAlign': 'center'}),\
        html.Div(
                    [
                        html.Div(id='secondpass_init', children=""),
                        
                    ]
        ),\
        dbc.Row(
            [
             dbc.Col(
                [
                     html.H5('''After running the first pass for demixing, specify a superpixel threshold here to identify any remaining signals''', style={'textAlign':'center'}),\
                ],\
                 width = 12
             ),\
            ]
        ),\
        dbc.Row(
           [
               dbc.Col(
                [ 
                   dash.dcc.Slider(id='superpixel_slider_secondpass',min=0.00,max=0.999,marks={0:'0', 0.1:'0.1', 0.2:'0.2', 0.3:'0.3', 0.4:'0.4', 0.5:'0.5', 0.6:'0.6',0.7:'0.7', 0.8:'0.8', 0.9:'0.9', 1:'1'},updatemode='mouseup',step=0.01,\
                                     value=0.9),\
                    html.H5("Superpixel Threshold", id="Slider Label_secondpass", style={'textAlign': 'center'}),\
                ],\
                   width=12
               ),\
           ]
        ),\
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Button(id="button_id_demix_secondpass", children="Run Demixing", style={'verticalalign':'center'}),\
                    ],\
                    width={"size": 6, "offset": 4},
                ),\
            ],\
        ),\

        html.Br(id='boolean-switch-demix'),\
        dcc.Download(id="download_demixing_results")
    ],
    style=SIDEBAR_STYLE,
)



app.layout = html.Div(
    # [html.H1("File Selected: None"), html.Div(controls), html.Div(id="folder-files")]
    [dbc.Row(
        html.H1("maskNMF Data Processing Dashboard", style={'textAlign': 'center'})
    ),\
     dbc.Row(
        [
            dbc.Col(html.Div([sidebar]), width=3),\
            dbc.Col(
            [
                dcc.Graph(
                    id='example-graph',
                    figure=fig_mc_pmd_plots
                ),\
                dcc.Graph(
                    id='trace_vis',
                    figure=fig_trace_vis
                ),\
                dash.dcc.Slider(id='pmd_mc_slider',min=0,max=100,marks=None,updatemode='mouseup',step=1,\
                             value=np.argmin(np.abs(100-1)))
            ], width = 9),\
        ],\
        align="center"
    ),\
     dbc.Row(
    [
         
         dcc.Download(id="download_elt")
    
    ]
    ),\
     

     ### Demixing ### 
     dbc.Row(
        [      
            dbc.Col(html.Div([sidebar_demixing]), width=3),\
            dbc.Col(
                [
                     dcc.Graph(
                        id='local_correlation_plot',
                        figure=fig_local_corr
                    ),\
                    dcc.Graph(
                        id='local_correlation_plot_secondpass',
                        figure=fig_local_corr_secondpass
                    ),\
                    html.Div(id='placeholder_local_corr_plot', children=""),\
                ],\
                width=3
            ),\
            dbc.Col(
                [
                    dcc.Graph(
                            id='local_pixel_corr_plot',
                            figure=fig_pixel_corr
                    ),\
                    dcc.Graph(
                            id='local_pixel_corr_plot_secondpass',
                            figure=fig_pixel_corr_secondpass
                    ),\
                ],\
                width=3
            ),\
            
            dbc.Col(
                [
                    dcc.Graph(
                        id='superpixel_plot',
                        figure=fig_superpixel
                    ),\
                    
                    dcc.Graph(
                        id='superpixel_plot_secondpass',
                        figure=fig_superpixel_secondpass
                    ),\
                ],\
                width=3
            ),\
            

        ],
        align="center"
    ),\
    dbc.Row(
        [
            dbc.Col(
                [
                    dcc.Graph(
                        id='post_demixing_summary_image',
                        figure=fig_post_demixing_summary_image
                    ),\
                    html.Div(id='placeholder_post_demixing', children=""),\
                ],\
                width=3
            ),\

            dbc.Col(
                [
                    dcc.Graph(
                        id='post_demixing_pixelwise_traces',
                        figure=fig_post_demixing_pixewise_traces
                    ),\
                ],\
                width=9
            ),\
        ]
    ),\
    ]
)





#######
## First init callback logic
#######

@app.callback(
    Output("superpixel_slider", "disabled"),
    Input("masknmf_flag", "on")
)
def switch_superpixel_slider_enable(flag):
    
    #In this case, masknmf_flag is ON, so the superpixel slider should be disabled
    if flag: 
        cache['first_pass_init_method'] = 'masknmf'
        return True
    #In this case, masknmf_flag is OFF, so the superpixel slider should be enabled
    else:
        cache['first_pass_init_method'] = 'superpixel'
        return False
        

#######
### Post init callback logic
#######

def get_a_halos(a):
    '''
    Params: 
        a: np.ndarray. Shape d1, d2, K (3 dimensions), where d1, d2 are the field of view dimensions, and K is the number of neural signals
    
    '''
    a_halos = np.zeros_like(a)
    for k in range(a.shape[2]):
        a_halos[:, :, k]= scipy.ndimage.distance_transform_bf(scipy.ndimage.distance_transform_bf(a[:, :, k]))

    return a_halos == 1



@app.callback(
    Output("post_demixing_summary_image", "figure"),
    Input("placeholder_post_demixing", "children"),
    prevent_initial_call=True
)
def generate_post_demixing_results(children):
    if cache['PMD_flag']:
        fin_rlt = cache['demixing_results']
        a, c, data_order, data_shape = fin_rlt['a'], fin_rlt['c'], fin_rlt['data_order'], fin_rlt['data_shape']
        # ac_mean_image = a.dot(np.mean(c.T, axis = 1, keepdims=True))
        # ac_mean_image = ac_mean_image.reshape((data_shape[0], data_shape[1]), order = data_order)
        background_image = cache['mean_img']
        
        a = a.reshape((data_shape[0], data_shape[1], -1), order = data_order)
        a_halos = get_a_halos(a)
    
        new_figure = px.imshow(background_image)
        
        for k in range(a.shape[2]):
            curr_img = a_halos[:, :, k]
            y,x = curr_img.nonzero()
            new_figure.add_scatter(
                x=x,
                y=y,
                mode='markers', 
                name="Signal {}".format(k+1),
                hoveron="points+fills",
            )
        new_figure.update_coloraxes(showscale=False)
        new_figure.update(layout_coloraxis_showscale=False)
        new_figure.update_layout(title_text="Click any pixel to see demixing", title_x=0.5)
        return new_figure
    else:
        return dash.no_update

@app.callback(
    Output("post_demixing_pixelwise_traces", "figure"),
    Input("post_demixing_summary_image", "clickData"),
    prevent_initial_call=True
)
def plot_demixing_result(clickData):
    if cache['PMD_object'] is not None:
        my_pmd_object = cache['PMD_object']
        if my_pmd_object.a is not None and my_pmd_object.c is not None: 
            a = scipy.sparse.csr_matrix(my_pmd_object.a)
            c = my_pmd_object.c
            x, y = get_points(clickData)
            if y >= cache['shape'][0] or x >= cache['shape'][1]: 
                raise ValueError("the click was out of bounds")
            temp_mat = np.arange(my_pmd_object.shape[0] * my_pmd_object.shape[1])
            temp_mat = temp_mat.reshape((my_pmd_object.shape[0], my_pmd_object.shape[1]), order=my_pmd_object.data_order)
            desired_index = temp_mat[y, x] ##Note y, x not x,y because the clickback returns the height as the second coordinate
        
            
            background_row = my_pmd_object.get_background_row(desired_index).cpu().numpy().flatten() + my_pmd_object.b[desired_index, :].flatten()
            
            ##Get the PMD Row
            desired_row = my_pmd_object.get_PMD_row(desired_index).cpu().numpy().flatten()

            ##Get the AC Row
            AC_trace = a.getrow(desired_index).dot(c.T)
            AC_trace = AC_trace.flatten()

            
            
            input_dict = {"Timesteps": [i for i in range(1, len(desired_row) + 1)], "PMD": desired_row, "Signal": AC_trace, "Background": background_row}
            trace_df = pd.DataFrame.from_dict(input_dict)

            fig_trace_vis = px.line(trace_df, x='Timesteps', y=trace_df.columns[1:], 
                color_discrete_map={
                     "PMD": "#353739",
                     # "Signal": "#fa8d22"
                 })
            fig_trace_vis.update_layout(title_text="Demixing at pixel height = {} width = {}".format(y, x), title_x=0.5)

            return fig_trace_vis
            
            
        else:
            return dash.no_update
    else:
        return dash.no_update
    
    
    


@app.callback(
    Output("local_pixel_corr_plot", "figure"),
    Input("local_correlation_plot", "clickData"),
    Input("local_correlation_plot", "figure"),
    prevent_initial_call=True
)
def update_single_pixel_corr_plot(clickData, local_corr_fig):
    #compute_single_pixel_correlation_image(row_index):
    if cache['PMD_object'] is None:
        return dash.no_update
    
    my_pmd_object = cache['PMD_object']
    button_clicked = list(ctx.triggered_prop_ids.keys())[0]
    
    if button_clicked == "local_correlation_plot.clickData":
        x, y = get_points(clickData)
        if cache['PMD_flag']:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

            temp_mat = np.arange(my_pmd_object.shape[0] * my_pmd_object.shape[1])
            temp_mat = temp_mat.reshape((my_pmd_object.shape[0], my_pmd_object.shape[1]), order=my_pmd_object.data_order)
            desired_index = temp_mat[y, x] ##Note y, x not x,y because the clickback returns the height as the second coordinate

            final_image = my_pmd_object.compute_single_pixel_correlation_image(desired_index, residual=False).cpu().numpy()
            final_image = final_image.reshape((my_pmd_object.shape[0], my_pmd_object.shape[1]), order=my_pmd_object.data_order)

            curr_fig = px.imshow(final_image, zmin=0, zmax=1)
            curr_fig.update_coloraxes(showscale=False)
            curr_fig.update(layout_coloraxis_showscale=False)
            curr_fig.update_yaxes(showticklabels=False)
            curr_fig.update_layout(title_text = "Pixel Corr. Image at ({}, {})".format(y,x),title_x=0.5)
            return curr_fig
        else:
            return dash.no_update
    
    elif button_clicked== "local_correlation_plot.figure":        
        if cache['PMD_flag']:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
                
            #Step 1: Get the local correlation figure
            my_img = np.array(local_corr_fig['data'][0]['z'])

            #Step 2: Get the indices of the "brightest" value
            x,y = np.unravel_index(my_img.argmax(), my_img.shape)

            #Step 3: Calculate the single pixel corr of this "brightest" value
            temp_mat = np.arange(my_pmd_object.shape[0] * my_pmd_object.shape[1])
            temp_mat = temp_mat.reshape((my_pmd_object.shape[0], my_pmd_object.shape[1]), order=my_pmd_object.data_order)
            desired_index = temp_mat[x, y] ##Note y, x not x,y because the clickback returns the height as the second coordinate

            final_image = my_pmd_object.compute_single_pixel_correlation_image(desired_index, residual=False).cpu().numpy()
            final_image = final_image.reshape((my_pmd_object.shape[0], my_pmd_object.shape[1]), order=my_pmd_object.data_order)

            curr_fig = px.imshow(final_image, zmin=0, zmax=1)
            curr_fig.update_coloraxes(showscale=False)
            curr_fig.update(layout_coloraxis_showscale=False)
            curr_fig.update_yaxes(showticklabels=False)
            curr_fig.update_layout(title_text = "Corr. Image for pixel at ({},{})".format(y,x),title_x=0.5)
            return curr_fig
            
        else:
            return dash.no_update
    
    

### CALLBACKS for CORR img clicking ###
@app.callback(
    Output("local_correlation_plot", "figure"),
    Output("superpixel_slider", "value"),
    Input("local_correlation_plot", "figure"), 
    Input("placeholder_local_corr_plot", "children"),
    prevent_initial_call=True
)
def compute_local_corr_values_and_set_superpixel_value(curr_fig, value):
    if cache['PMD_flag']:
        var_img = cache['noise_var_img']
        curr_fig = px.imshow(var_img.squeeze())
        curr_fig.update_layout(title_text = "Noise Variance Image".format(value),title_x=0.5)
        
        curr_fig.update_coloraxes(showscale=False)
        curr_fig.update(layout_coloraxis_showscale=False)
        
        #Finally pick the init superpixel value
        superpixel_threshold = cache['localnmf_params']['superpixels_corr_thr'][0]
        return curr_fig, superpixel_threshold
    
    else:
        return dash.no_update    




### CALLBACKS for superpixel clicking ###
@app.callback(
    Output("superpixel_plot", "figure"),
    Input("superpixel_plot", "figure"), 
    Input("superpixel_slider", "value"),
    Input("superpixel_slider", "disabled"),
    prevent_initial_call=True,
)
def generate_superpixel_plot_firstpass(curr_fig, value, disabled_flag):
    '''
    TODO: 
    Read parameters in a more principled way -- this boilerplate code is extremely hard to maintain
    '''
    if disabled_flag:
        if cache['PMD_flag']:
            zeros_element = np.zeros((cache['shape'][0], cache['shape'][1]))
            curr_fig = px.imshow(zeros_element)
        else:
            zeros_element = np.zeros(default_figsize)
            curr_fig = px.imshow(zeros_element)
        
        curr_fig = curr_fig.update_layout(title_text="No superpixels - using maskNMF instead")
        curr_fig.update_yaxes(showticklabels=False)
        return curr_fig
    elif cache['PMD_flag']:
        
        if cache['PMD_object'].a_init is not None: 
            print("IN THE KEY LOOP")
            if torch.cuda.is_available():
                device='cuda'
            else:
                device = 'cpu'
            my_pmd_object = cache['PMD_object']
            my_pmd_object.reset()
            cache['PMD_object'] = my_pmd_object
            
            
        
        lnmf_params = cache['localnmf_params']
        
        length_cut=lnmf_params['length_cut'][0] 
        th= lnmf_params['th'][0] 

        ##Do not need to modify
        residual_cut = lnmf_params['residual_cut'][0]
        num_plane=lnmf_params['num_plane']
        patch_size= lnmf_params['patch_size']
        plot_en = True
        text=True
        pseudo_2 = lnmf_params['pseudo_2'][0]


        cut_off_point = value
        my_pmd_object = cache['PMD_object']
        if my_pmd_object.a is not None or my_pmd_object.c is not None: 
            my_pmd_object.reset()
        my_pmd_object.initialize_signals_superpixels(num_plane, cut_off_point, residual_cut, length_cut, th, pseudo_2, \
                                       text =True, plot_en = True)
        superpixel_image = my_pmd_object.superpixel_image_recent
        
        cache['PMD_object'] = my_pmd_object
        curr_fig = px.imshow(superpixel_image)
        curr_fig.update_yaxes(showticklabels=False)
        curr_fig.update_layout(title_text = "Superpixels Image, threshold = {}".format(value),title_x=0.5)
        
        ##Update the value
        lnmf_params = cache['localnmf_params']
        lnmf_params['superpixels_corr_thr'][0] = value
        cache['localnmf_params'] = lnmf_params
        return curr_fig
    
    else:
        return dash.no_update
    


### CALLBACKS for point-based clicking

def get_points(clickData):
    if not clickData:
        raise dash.exceptions.PreventUpdate
    outputs = {k: clickData["points"][0][k] for k in ["x", "y"]}
    return int(outputs["x"]), int(outputs["y"])

@app.callback(
    Output("trace_vis", "figure"),
    Input("example-graph", "clickData"),
)
def click(clickData):
    x, y = get_points(clickData)
    if cache['PMD_flag']:
        temp_mat = np.arange(cache['shape'][0] * cache['shape'][1])
        temp_mat = temp_mat.reshape((cache['shape'][0], cache['shape'][1]), order=cache['order'])
        desired_index = temp_mat[y, x] ##Note y, x not x,y because the clickback returns the height as the second coordinate
        
        desired_row = (cache['U'].getrow(desired_index).dot(cache['R'])).dot(cache['V']).flatten()
        
        trace = desired_row.flatten()
        trace = pd.DataFrame(trace, columns = ['X'])
        fig_trace_vis = px.line(trace, y="X", 
                       labels={
                     "index": "Frame Number",
                     "X": "A.U.",
                 },)
        fig_trace_vis.update_layout(title_text="PMD Trace of pixel height = {} width = {}".format(y, x), title_x=0.5)
        
        return fig_trace_vis

    else:
        return dash.no_update


### CALLBACKS FOR SCROLLBAR
def load_mc_frame(index):
    data = np.array(tifffile.imread(cache['navigated_file'], key=index))
    return data

def get_PMD_frame(index):
    RV = cache['R'].dot(cache['V'][:, [index]])
    URV = cache['U'].dot(RV)
    

    URV = URV.reshape(cache['shape'][:2], order=cache['order'])
    URV *= cache['noise_var_img']
    URV += cache['mean_img']
    URV -= np.amin(URV)
    return URV
        
    
@app.callback(Output('example-graph', 'figure'), Input('example-graph', 'figure'), Input("pmd_mc_slider", "value"))
def update_motion_image(curr_fig, value):
    if cache['navigated_file'] == cache['no_file_flag']:
        return dash.no_update
    else:
        
        min_val, max_val = (0, cache['shape'][2]-1)
        value = max(min_val, value)
        value = min(max_val, value)
        used_data = [load_mc_frame(value), get_PMD_frame(value), cache['noise_var_img']]
        
        max_val = max(np.amax(used_data[0]), np.amax(used_data[1]))
        
        if np.amax(used_data[2]) != 0:
            final_noise_var_img = max_val/(np.amax(used_data[2])) * used_data[2]
            used_data[2] = final_noise_var_img


        num_imgs = 3 ##HARDCODED FOR NOW, CHANGE IF NEEDED
        for i in range(3):
            curr_fig['data'][i]['z'] = used_data[i]

        img_name_list = ["Raw Frame {}".format(value+1), "Registered + PMD-denoised Frame {}".format(value+1), "Scaled Noise Variance Image Frame {}".format(value+1)]
        for i, name in enumerate(img_name_list):
            curr_fig['layout']['annotations'][i]['text'] = img_name_list[i]

        return curr_fig

    
    



### CALLBACKS FOR FILE INPUT
def get_last_folder_val(folder_string):
    return folder_string.split("/")[-1]

@app.callback(Output("dropdown", "value"), Output("folder-files", "children"), Output("curr_folder", "children"), Output("dropdown", "options"), Input("dropdown", "value"))
def list_all_files(folder_name):

    #Decide if input is a file or not:
    
    folder_response = "Folder: {}"
    default_value = ""
    
    decided_path = os.path.normpath(os.path.join(cache['navigated_folder'], folder_name))
    is_file = os.path.isfile(decided_path)
    is_dir = os.path.isdir(decided_path)

    if is_file:
        #A file has been selected, need to update texts: 
        new_text = "File Selected: {}".format(folder_name)
        current_folder = cache['navigated_folder']#present_dir[0]
        # present_dir[1] = decided_path
        cache['navigated_file'] = decided_path
        final_dir = [{"label": x, "value": x} for x in list_dir(current_folder)]
        return default_value, new_text, folder_response.format(get_last_folder_val(current_folder)), final_dir
    elif is_dir:
        cache['navigated_file'] = cache['no_file_flag']
        cache['navigated_folder'] = decided_path
        final_dir = [{"label": x, "value": x} for x in list_dir(decided_path)]
        return default_value, "File Selected: None", folder_response.format(get_last_folder_val(cache['navigated_folder'])), final_dir
    else:
        raise ValueError("Invalid suggestion")

        
##########
## MOTION CORRECTION + PMD COMPRESSION CALLBACKS
##########

@dash.callback(
    Output("block_pmd_placeholder_1", "children"),
    Input("block_pmd_1", "value")
)
def update_pmd_blocksize_height(value):
    new_pmd_params = cache['pmd_params']
    new_pmd_params['block_height'] = value
    cache['pmd_params'] = new_pmd_params
    
    return dash.no_update


@dash.callback(
    Output("block_pmd_placeholder_2", "children"),
    Input("block_pmd_2", "value")
)
def update_pmd_blocksize_width(value):
    new_pmd_params = cache['pmd_params']
    new_pmd_params['block_width'] = value
    cache['pmd_params'] = new_pmd_params
    
    return dash.no_update

@dash.callback(
    Output("boolean-switch-dumboutput","children"), Input("register_flag", "on")
)
def change_register_flag(new_mode):
    print("the new mode of the register flag is {}".format(new_mode))
    cache['register_run_flag'] = new_mode
    
    return dash.no_update


@dash.callback(
    Output("placeholder", "children"), Output("pmd_mc_slider", "value"), Output("download_elt", "data"), Output("placeholder_local_corr_plot", "children"), Output("pmd_mc_slider", "max"),
    inputs=Input("button_id", "n_clicks"),
    background=True,
    manager=background_callback_manager,
    running=[
        (Output("button_id", "disabled"), True, False),
        (
            Output("placeholder", "style"),
            {"visibility": "hidden"},
            {"visibility": "visible"},
        ),
    ],
    prevent_initial_call=True
)
def register_and_compress_data(n_clicks):
    data_folder = cache['navigated_folder'] 
    input_file = cache['navigated_file'] 

    from datetime import datetime
    import os

    def get_shape(filename):
        import tifffile
        with tifffile.TiffFile(filename) as tffl:
          num_frames = len(tffl.pages)
          for page in tffl.pages[0:1]:
              image = page.asarray()
              x, y = page.shape
        return (x,y,num_frames)

    def get_file_name(filestring):
        splitted_values = filestring.split("/")[-1]
        name_value = splitted_values.split(".")[0]
        return name_value

    def define_new_folder(filestring):
        filename = get_file_name(filestring)
        now = datetime.now()

        # dd/mm/YY H:M:S
        dt_string = now.strftime(filename + "_DASH_results_%d_%m_%Y_%H_%M_%S")
        print(dt_string)	
        return dt_string


    def set_and_create_folder_path(input_file, data_folder):
        new_folder_name = define_new_folder(input_file)
        final_path = os.path.join(data_folder, new_folder_name)
        if not os.path.exists(final_path):
            os.mkdir(final_path)
        print("creating folder at location {}".format(final_path))
        return final_path


    #NOTE: this data folder will also contain the location of the TestData
    data_folder = set_and_create_folder_path(cache['navigated_file'], cache['navigated_folder'])
    cache['save_folder'] = data_folder
    input_file = cache['navigated_file']#present_dir[1]

    mc_params_dict = cache['mc_params']

    if cache['register_run_flag']:
        register = True
    else:
        register = False
    devel = mc_params_dict['devel']

    dx = mc_params_dict['dx'] 
    dy = mc_params_dict['dy'] 

    dxy = (dx, dy)

    max_shift_in_um_xdimension = mc_params_dict['max_shift_in_um'][0] 
    max_shift_in_um_ydimension = mc_params_dict['max_shift_in_um'][1] 

    max_shift_um = (max_shift_in_um_xdimension, max_shift_in_um_ydimension)

    max_deviation_rigid = mc_params_dict['max_deviation_rigid']

    patch_motion_um_x = mc_params_dict['patch_motion_um'][0] 
    patch_motion_um_y = mc_params_dict['patch_motion_um'][1] 

    patch_motion_um = (patch_motion_um_x, patch_motion_um_y)

    overlaps_x = mc_params_dict['overlaps'][0] 
    overlaps_y = mc_params_dict['overlaps'][1] 

    overlaps = (overlaps_x, overlaps_y)

    border_nan = 'copy'

    niter_rig = mc_params_dict['niter_rig'] #2 #@param {type:"slider", min:1, max:10, step:1}
    niter_els = mc_params_dict['niter_els'] #2 #@param {type:"slider", min:1, max:10, step:1}

    pw_rigid = mc_params_dict['pw_rigid'] #True #@param {type:"boolean"}

    use_gSig_filt = mc_params_dict['use_gSig_filt'] #False #@param {type:"boolean"}
    gSig_filt_x = mc_params_dict['gSig_filt'] #3 #@param {type:"slider", min:0, max:30, step:1}
    gSig_filt_y = mc_params_dict['gSig_filt'] #3 #@param {type:"slider", min:0, max:30, step:1}
    
    sketch_template = True

    if use_gSig_filt: 
        gSig_filt = (gSig_filt_x, gSig_filt_y)
    else:
        gSig_filt = None

    frames_per_split =  500

    INPUT_PARAMS = {
        # Caiman Internal:
        'local': {'devel': devel},
        'caiman': {'dxy': dxy,
                   'max_shift_um': max_shift_um,
                   'max_deviation_rigid': max_deviation_rigid,
                   'patch_motion_um': patch_motion_um,
                   'overlaps': overlaps,
                   'border_nan': 'copy',
                   'niter_rig': niter_rig,
                   'niter_els': niter_els,
                   'pw_rigid': pw_rigid,
                   'gSig_filt': gSig_filt,
                   'splits' : frames_per_split,
                   'sketch_template': True},
    }


    import multiprocessing
    import os
    import shutil
    import pathlib
    import sys
    import math
    import glob

    import numpy as np
    import tifffile

    import datetime



    from tqdm import tqdm

    import yaml



    def display(msg):
        """
        Printing utility that logs time and flushes.
        """
        tag = '[' + datetime.datetime.today().strftime('%y-%m-%d %H:%M:%S') + ']: '
        sys.stdout.write(tag + msg + '\n')
        sys.stdout.flush()

    def parinit():
        """
        Initializer run by each process in multiprocessing pool.
        """
        os.environ['MKL_NUM_THREADS'] = "1"
        os.environ['OMP_NUM_THREADS'] = "1"
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        num_cpu = multiprocessing.cpu_count()
        os.system('taskset -cp 0-%d %s > /dev/null' % (num_cpu, os.getpid()))


    def runpar(function, arguments, nproc=None, **kwargs):
        """
        Maps an input function to a sequence of arguments.
        """
        if nproc is None:
            nproc = multiprocessing.cpu_count()
        with multiprocessing.Pool(initializer=parinit, processes=nproc) as pool:
            res = pool.map(functools.partial(function, **kwargs), arguments)
        pool.join()
        return res


    def flatten(mapping):
        """
        Flattens a nested dictionary assuming that there are no key collisions.
        """
        items = []
        for key, val in mapping.items():
            if isinstance(val, dict):
                items.extend(flatten(val).items())
            else:
                items.append((key, val))
        return dict(items)


    def load_config(default):
        """
        Loads user-provided yaml file containing parameters key-value pairs.
        Omitted optional parameter keys are filled with default values.
        Parameters
        ----------
        filename : string
            Full path + name of config 'yaml' file.
        Returns
        -------
        params : dict
            Parameter key-value pairs.
        """

        params = dict()
        # Insert default values for missing optional fields
        display('Inserting defaults for missing optional arguments')
        for group, group_params in default.items():
            display(f"Using all defaults in group '{group}={group_params}'")
            params[group] = group_params
        display("Config file successfully loaded.")
        return flatten(params)


    def write_params(filename, required={}, default={}, **params):
        """
        Writes verbose parameter dictionary containing all default, modified, and
        simulated fields to a specified output location.
        Parameters
        ----------
        filename : string
            Full path + name for destination of output config file.
        params : dict
            User-provided parameter key-value pairs to be written.
        Returns
        -------
        None :
        """

        # Construct Mapping Of Keys -> Categories
        reverse_map = {}
        for group, keys in required.items():
            for key in keys:
                reverse_map[key] = group
        for group, group_params in default.items():
            for key, _ in group_params.items():
                reverse_map[key] = group

        # Undo Flattening
        grouped_params = {'unused': {}}
        for group, _ in required.items():
            grouped_params[group] = {}
        for group, _ in default.items():
            grouped_params[group] = {}
        for key, val in params.items():
            try:
                grouped_params[reverse_map[key]][key] = val
            except KeyError:
                grouped_params['unused'][key] = val

        # Write Output
        display(f"Writing verbose config to ({filename})...")
        with open(filename, 'w') as stream:
            yaml.dump(grouped_params, stream, default_flow_style=False)
        display("Verbose config written successfully.")






    DATA_WRITERS = {
        'tiff': tifffile.imwrite,
        'tif': tifffile.imwrite
    }
    VALID_EXTS = list(DATA_WRITERS.keys())
    CONFIG_NAME = 'config.yaml'


    def get_caiman_memmap_shape(filename):
        fn_without_path = os.path.split(filename)[-1]
        fpart = fn_without_path.split('_')[1:-1]  # The filename encodes the structure of the map
        d1, d2, d3, T, order = int(fpart[-9]), int(fpart[-7]), int(fpart[-5]), int(fpart[-1]), fpart[-3]
        return (d1, d2, T)


    def write_output_simple(targets, out_file, batch_size = 1000, dtype = np.float64):   
        from jnormcorre.utils.movies import load
        with tifffile.TiffWriter(out_file, bigtiff=True) as tffw:
            for index in tqdm(range(len(targets))):
                file = targets[index]
                file_split = file.rsplit(".", maxsplit=1)
                shape = get_caiman_memmap_shape(file)
                num_iters = math.ceil(shape[2] / batch_size)
                for k in range(num_iters):
                    start = k*batch_size
                    end = min((k+1)*batch_size, shape[2])
                    data = load(file, subindices=range(start, end)).astype(dtype)
                    for j in range(min(end - start, batch_size)):
                        tffw.write(data[j, :, :], contiguous=True)           


    def chunk_singlepage_data(filename, batch_size = 10000):
        '''
        Saves singlepage tiff files as a sequence of multipage tifs
        Code adapted from @JakeHeffley

        Params: 
            filename: str. String describing the path to the data file
            batch_size: int. Number of frames to save in each multipage tif
        '''

        filename_list = []
        total_count = 0

        with tifffile.TiffFile(filename) as tffl:
            T = tffl.series[0].shape[0] #total number of frames in original movie
            batch_size = T
            iters = math.ceil(T/batch_size)


            full_movie = tffl.asarray(out='memmap')

            for i in tqdm(range(iters)):
                start = i * batch_size
                end = min((i+1) * (batch_size), T)

                fileparts = filename.rsplit(".", maxsplit=1)
                this_batch = full_movie[start:end, :, :]

                #Add filename to name list
                new_filename = os.path.join(fileparts[0] + "_" + str(total_count) + "." + fileparts[1])
                tifffile.imsave(new_filename, this_batch)
                filename_list.append(new_filename)
                total_count += 1

        #As last step, delete original data
        return filename_list

    def delete_targets(targets):
        display("DELETING TARGETS")
        for file in targets:
            os.remove(file)


    def resolve_dataformats(filename):
        '''
        Function for managing bad data formats (such as single-page tif files) which are tough to load. Resolves these issues by loading the data into memmap format and then saving the data (in small batches) into a better format
        Input: 
            filename: str. String describing the full filepath of the datafile
        Returns: 
            file_output: list of strings. In this list, each string is a filename. These files, taken together, form the entire dataset
        '''
        _, extension = os.path.splitext(filename)[:2]
        if extension in ['.tif', '.tiff', '.btf']:  # load tif file
            with tifffile.TiffFile(filename) as tffl:
                multi_page = True if tffl.series[0].shape[0] > 1 else False
                if len(tffl.pages) == 1:
                    display("Data is saved as single page tiff file. We will re-save data as sequence of smaller tifs to improve performance, but this will take time. To avoid this issue, save your data as multi-page tiff files")
                    file_output = chunk_singlepage_data(filename)
                    return file_output

        file_output = [filename]
        return file_output


    def motion_correct(filename,
                       outdir,
                       dxy = (2., 2.),
                       max_shift_um = (12., 12.),
                       max_deviation_rigid = 3,
                       patch_motion_um = (100., 100.),
                       overlaps = (24, 24),
                       border_nan= 'copy',
                       niter_rig = 4,
                       splits=200,
                       pw_rigid = True,
                       gSig_filt=None,
                       save_movie=True,
                       dtype='int16',
                       sketch_template=False,
                       **params):
        """
        Runs motion correction from caiman on the input dataset with the
        option to process the same dataset in multiple passes.
        Parameters
        ----------
        filename : string
            Full path + name for destination of output config file.
        outdir : string
            Full path to location where outputs should be written.
        dxy: tuple (2 elements)
            Spatial resolution in x and y in (um per pixel)
        max_shift_um: tuple (2 elements)
            Maximum shift in um
        max_deviation_rigid: int
            Maximum deviation allowed for patch with respect to rigid shifts
        patch_motion_um: 
            Patch size for non rigid correction in um
        overlaps:
            Overlap between patches
        border_nan: 
            See linked caiman docs for details
        niter_rig: int
            Number of passes of rigid motion correction (used to estimate template)
        splits: int
            We divide the registration into chunks (temporally). Splits = number of frames in each chunk. So splits = 200 means we break the data into chunks, each containing ~200 frames.
        pw_rigid: boolean 
            Indicates whether or not to run piecewise rigid motion correction
        devel: boolean
            Indicates whether this code is run in development mode. If in development mode, the original data is not deleted.
        Returns
        -------
        None :
        """

        from jnormcorre.utils.movies import load
        from jnormcorre import motion_correction
        import math

        # Iteratively Run MC On Input File
        display("Running motion correction...")
        target = resolve_dataformats(filename)

        total_frames_firstfile = get_shape(target[0])[2]
        splits = math.ceil(total_frames_firstfile / splits)
        display("Number of chunks is {}".format(splits))

        # Since running on colab, which has only 2 vCPU, don't use multiprocessing, use GPU or TPU since the main compute is implemented in jax
        dview=None


        # Default MC_dict
        mc_dict = {
        'border_nan': 'copy',               # flag for allowing NaN in the boundaries
        'max_deviation_rigid': 3,           # maximum deviation between rigid and non-rigid
        'max_shifts': (6, 6),               # maximum shifts per dimension (in pixels)
        'min_mov': -5,                      # minimum value of movie
        'niter_rig': 4,                     # number of iterations rigid motion correction
        'niter_els': 1,                     # number of iterations of piecewise rigid motion correction
        'nonneg_movie': True,               # flag for producing a non-negative movie
        'num_frames_split': 80,             # split across time every x frames
        'num_splits_to_process_els': None,  # The number of splits of the data which we use to estimate the template for the rigid motion correction. If none, we look at entire dataset.
        'num_splits_to_process_rig': None,  # The number of splits of the data which we use to estimate the template for pwrigid motion correction. If none, we look at entire dataset.
        'overlaps': (32, 32),               # overlap between patches in pw-rigid motion correction
        'pw_rigid': False,                  # flag for performing pw-rigid motion correction
        'shifts_opencv': True,              # flag for applying shifts using cubic interpolation (otherwise FFT)
        'splits_els': 14,                   # number of splits across time for pw-rigid registration
        'splits_rig': 14,                   # number of splits across time for rigid registration
        'strides': (96, 96),                # how often to start a new patch in pw-rigid registration
        'upsample_factor_grid': 4,          # motion field upsampling factor during FFT shifts
        'indices': (slice(None), slice(None)),  # part of FOV to be corrected
        'gSig_filt': None
    }

        max_shifts = [int(a/b) for a, b in zip(max_shift_um, dxy)]
        strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])

        mc_dict['pw_rigid']= pw_rigid
        mc_dict['strides'] = strides
        mc_dict['overlaps'] = overlaps
        mc_dict['max_deviation_rigid'] = max_deviation_rigid
        mc_dict['border_nan'] = 'copy'
        mc_dict['niter_rig'] = niter_rig
        mc_dict['niter_els'] = niter_els
        if sketch_template:
            mc_dict['num_splits_to_process_els'] = 5
            mc_dict['num_splits_to_process_rig'] = 5
        mc_dict['gSig_filt'] = gSig_filt
        mc_dict['max_shifts'] = max_shifts
        mc_dict['splits_els'] = splits
        mc_dict['splits_rig'] = splits


        corrector = motion_correction.MotionCorrect(target, dview=dview, **mc_dict)

        # Run MC, Always Saving Non-Final Outputs For Use In Next Iteration
        corrector_obj = corrector.motion_correct(
            save_movie=False
        )


        display("Motion correction completed.")


        # Save Frame-wise Shifts
        display(f"Saving computed shifts to ({outdir})...")
        np.savez(os.path.join(outdir, "shifts.npz"),
                 shifts_rig=corrector.shifts_rig,
                 x_shifts_els=corrector.x_shifts_els if pw_rigid else None,
                 y_shifts_els=corrector.y_shifts_els if pw_rigid else None)
        display('Shifts saved as "shifts.npz".')

        corrector_obj.batching=10 ##Long term need to avoid this...
        return corrector_obj, target


    # Get data and config filenames from cmd line args
    data_name = input_file
    outdir = cache['save_folder'] #results_output_folder[0]
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    # Load User-Provided Params
    params = load_config(INPUT_PARAMS)
    print(params)
    write_params(os.path.join(outdir, "MotionCorrectConfig.yaml"),
                  default=INPUT_PARAMS,
                  **params)

    # Run Single Pass Motion Correction
    try:
        if register:
            corrector, target = motion_correct(data_name, outdir, **params)
            data_name = target[0]
            input_file = data_name
        else:
            corrector = None
            data_name = resolve_dataformats(data_name)[0]
            input_file = data_name
            import jax
            jax.clear_backends()
            torch.cuda.empty_cache()
    except Exception as e:
        print(e)
        display("Program crashed.")
        import jax
        jax.clear_backends()

    ##########
    ##
    ##
    ## PMD Compression + Denoising portion
    ##
    ##
    ##########
  
    #NOTE: this data folder will also contain the location of the TestData
    data_folder = cache['save_folder'] 

    pmd_params_dict = cache['pmd_params']
    block_height = pmd_params_dict['block_height']
    block_width = pmd_params_dict['block_width'] 
    block_sizes = [block_height, block_width]

    overlaps_height = pmd_params_dict['overlaps_height'] 
    overlaps_width = pmd_params_dict['overlaps_width'] 

    if overlaps_height > block_height: 
        print("Overlaps height was set to be greater than block height, which is not valid")
        print("Setting overlaps to be 5")
        overlaps_height = 5

    if overlaps_width > block_width:
        print("Overlaps width was set to be greater than width height, which is not valid \
        Setting overlaps to be 5")
        overlaps_width = 5

    overlap = [overlaps_height, overlaps_width]


    window_length = pmd_params_dict['window_length'] 
    if window_length <= 0:
        print("Window length cannot be negative! Resetting to 6000")
        window_length = 6000
    start = 0
    end = window_length

    background_rank = pmd_params_dict['background_rank'] 
    deconvolve=True
    deconv_batch=1000

    ###THESE PARAMS ARE NOT MODIFIED
    sim_conf = 5
    max_rank_per_block = 40 

    #@markdown Keep run_deconv true unless you do not want to run maskNMF demixing
    run_deconv = True
    max_components = max_rank_per_block

    INPUT_PARAMS = {
        # Caiman Internal:
        'localmd':{'block_height':block_height,
        'block_width':block_width,
        'overlaps_height':overlaps_height,
        'overlaps_width':overlaps_width,
        'window_length':window_length,
        'background_rank':background_rank,
        'max_rank_per_block':max_rank_per_block,
        'run_deconv':run_deconv 
        }
    }

    block_sizes = block_sizes
    overlap = overlap

    def load_config(default):
        """
        Loads user-provided yaml file containing parameters key-value pairs.
        Omitted optional parameter keys are filled with default values.
        Parameters
        ----------
        filename : string
            Full path + name of config 'yaml' file.
        Returns
        -------
        params : dict
            Parameter key-value pairs.
        """

        params = dict()
        # Insert default values for missing optional fields
        display('Inserting defaults for missing optional arguments')
        for group, group_params in default.items():
            display(f"Using all defaults in group '{group}={group_params}'")
            params[group] = group_params
        display("Config file successfully loaded.")
        return flatten(params)


    def write_params(filename, required={}, default={}, **params):
        """
        Writes verbose parameter dictionary containing all default, modified, and
        simulated fields to a specified output location.
        Parameters
        ----------
        filename : string
            Full path + name for destination of output config file.
        params : dict
            User-provided parameter key-value pairs to be written.
        Returns
        -------
        None :
        """

        # Construct Mapping Of Keys -> Categories
        reverse_map = {}
        for group, keys in required.items():
            for key in keys:
                reverse_map[key] = group
        for group, group_params in default.items():
            for key, _ in group_params.items():
                reverse_map[key] = group

        # Undo Flattening
        grouped_params = {'unused': {}}
        for group, _ in required.items():
            grouped_params[group] = {}
        for group, _ in default.items():
            grouped_params[group] = {}
        for key, val in params.items():
            try:
                grouped_params[reverse_map[key]][key] = val
            except KeyError:
                grouped_params['unused'][key] = val

        # Write Output
        display(f"Writing verbose config to ({filename})...")
        with open(filename, 'w') as stream:
            yaml.dump(grouped_params, stream, default_flow_style=False)
        display("Verbose config written successfully.")


    def perform_localmd_pipeline(input_file, block_sizes, overlap, frame_range, background_rank, \
                            max_components, sim_conf, \
                             tiff_batch_size,deconv_batch, folder, run_deconv=True, pixel_batch_size=100,\
                            batching=5, dtype="float32",  order="F", corrector = None):

        from localmd.decomposition import localmd_decomposition, display
        import localmd.tiff_loader as tiff_loader
        import scipy
        import scipy.sparse
        import jax
        import jax.scipy
        import jax.numpy as jnp
        import numpy as np
        from jax import jit, vmap
        import functools
        from functools import partial
        import time
        import torch

        start, end = frame_range[0], frame_range[1]


        #Reshape U using order="F" here
        U, R, s, V, tiff_loader_obj = localmd_decomposition(input_file, block_sizes, overlap, [start, end], \
                                        max_components=max_components, background_rank = background_rank, sim_conf=sim_conf,\
                                         tiff_batch_size=tiff_batch_size,pixel_batch_size=pixel_batch_size, batching=batching, dtype=dtype, order=order, \
                                         num_workers=0, frame_corrector_obj = corrector)



        
            
        
        def save_decomposition(U, R, s, V, load_obj, folder, order="F"):
            '''
            Write results to temporary location 
            
            '''
            file_name = "decomposition.npz"
            final_path = os.path.join(folder, file_name)

            #Write to cache for quick access
            cache['U'] = U
            cache['order'] = order
            cache['R_orig'] = R
            cache['R'] = R * s[None, :]
            cache['s'] = s
            cache['V'] = V
            cache['shape'] = load_obj.shape
            cache['mean_img'] = tiff_loader_obj.mean_img
            cache['noise_var_img'] = tiff_loader_obj.std_img
            cache['PMD_flag'] = True
            
            ##Slowly work in the PMD Video to the app 
            if torch.cuda.is_available():
                device='cuda'
            else:
                device = 'cpu'
            cache['PMD_object'] = PMDVideo(U.tocsr(), R, s, V, cache['shape'], data_order=order, device=device)
            
            np.savez(final_path, fov_shape = load_obj.shape[:2], \
                fov_order=order, U_data = U.data, \
                U_indices = U.indices,\
                U_indptr=U.indptr, \
                U_shape = U.shape, \
                U_format = type(U), \
                R = R, \
                s = s, \
                Vt = V, \
                 mean_img = tiff_loader_obj.mean_img, \
                 noise_var_img = tiff_loader_obj.std_img)

            display("the decomposition.npz file is saved at {}".format(folder))


        ##Step 2i: Save the results: 
        save_decomposition(U.tocsr(), R, s, V, tiff_loader_obj, folder, order=order)



    tiff_batch_size = 500
    import jax
    import torch
    jax.clear_backends()
    torch.cuda.empty_cache()
    params = load_config(INPUT_PARAMS)
    print(params)
    write_params(os.path.join(outdir, "CompressionConfig.yaml"),
                default=INPUT_PARAMS,
                **params)

    import localmd
    from localmd.decomposition import threshold_heuristic
    from localmd import tiff_loader
    from jnormcorre.motion_correction import frame_corrector 
    from masknmf.engine.sparsify import get_factorized_projection
    pmdresults = perform_localmd_pipeline(input_file, block_sizes, overlap, [start, end], background_rank, \
                          max_components, sim_conf, tiff_batch_size,deconv_batch,data_folder, pixel_batch_size=100, run_deconv=run_deconv,\
                          batching=5, dtype="float32",  order="F", corrector = corrector)
    torch.cuda.empty_cache()
    jax.clear_backends() 

    downloaded_data_file = os.path.join(cache['save_folder'], "decomposition.npz")
    return None, 0, dash.no_update, " ", cache['shape'][2] 

        
def display(msg):
        """
        Printing utility that logs time and flushes.
        """
        tag = '[' + datetime.datetime.today().strftime('%y-%m-%d %H:%M:%S') + ']: '
        sys.stdout.write(tag + msg + '\n')
        sys.stdout.flush()


@dash.callback(
    Output("boolean-switch-demix","children"), Input("masknmf_flag", "on")
)
def change_init_settings(new_mode):
    localnmf_params = cache['localnmf_params']
    init_method = localnmf_params['init']
    if new_mode:
        init_method[0] = 'custom'
    else:
        init_method[0] = 'lnmf'
        
    localnmf_params['init'] = init_method
    cache['localnmf_params'] = localnmf_params
    
    return dash.no_update
        
        
#Eventually move this to another file (masknmf file)
def dense_init_pipeline():
    from masknmf.engine.sparsify import get_factorized_projection
    import torch
    import jax
    if torch.cuda.is_available():
        cpu_only=False
        device='cuda'
    else:
        cpu_only=True
        device = 'cpu'
    #Step 1 run deconvolution
    display("running temporal sparsening operation")
    deconv_batch = 1000
    deconv_components = get_factorized_projection(
          cache['U'],
          cache['R_orig'],
          cache['s'][:, None] * cache['V'],
          batch_size=deconv_batch, device=device).astype(np.float64)
    
    cache['deconvolved_temporal'] = deconv_components

    
    display('running masknmf dense init pipeline')
    confidence = 0.6
    allowed_overlap = 70
    

   
        
    block_dims_x = 20 
    block_dims_y = 20 
    frame_len = 100
    
    spatial_thresholds_1 = 0.7
    spatial_thresholds_2 = 0.7

    spatial_thresholds = [spatial_thresholds_1, spatial_thresholds_2]


    a = run_masknmf(cache['save_folder'], confidence, allowed_overlap, cpu_only,\
                block_dims_x, block_dims_y, frame_len, spatial_thresholds)
    
    if torch.cuda.is_available():
        device='cuda'
    else:
        device = 'cpu'
    if cache['PMD_object'] is not None:
        my_pmd_object = cache['PMD_object']
        my_pmd_object.reset()
        cache['PMD_object'] = my_pmd_object

    else:
        cache['PMD_object'] = PMDVideo(cache['U'].tocsr(), cache['R_orig'], cache['s'],\
                                   cache['V'], cache['shape'], data_order=cache['order'], device=device)
    
    my_pmd_object = cache['PMD_object']
    my_pmd_object.initialize_signals_custom({'a': a})
    cache['PMD_object'] = my_pmd_object

    return my_pmd_object.a

    
    

def run_masknmf(data_folder, confidence, allowed_overlap, cpu_only,\
                block_dims_x, block_dims_y, frame_len, spatial_thresholds):

    
    ## TODO: For organizational purposes move the imports to the top
    import torch
    import torch_sparse
    import sys

    import copy
    #Misc imports
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib as mpl
    import colorsys
    from matplotlib import patches,  lines
    from matplotlib.patches import Polygon
    import shutil
    import time
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage import io
    import skimage
    from skimage import measure
    from skimage import filters

    import random
    import numpy as np
    import os
    import skimage.io as io

    import torch_sparse

    import scipy
    import scipy.sparse


    import boto3
    from botocore.config import Config
    from botocore import UNSIGNED

    from masknmf.engine.segmentation import segment_local_UV, filter_components_UV
    from masknmf.detection.maskrcnn_detector import maskrcnn_detector


    U_sparse = cache['U']

    order = cache['order']
    shape = cache['shape']
    d1,d2 = shape[:2]
    T = shape[2]
    dims = (d1, d2, T)
    temporal_components = cache['deconvolved_temporal']
    mixing_weights = cache['R_orig']


    #This is where we create the neural network
    dir_path = "neuralnet_info" 

    #Specify where to save these outputs
    MASK_NMF_CONFIG = os.path.join(dir_path, "config_nn.yaml")
    MASK_NMF_MODEL = os.path.join(dir_path, "model_final.pth")

    #Specify where to retrieve the neural net data. In this case, in the apasarkar-public bucket on AWS S3
    bucket_loc = "apasarkar-public"
    config_file_name = "config.yaml"
    weights_file_name = "model_final.pth"



    if not os.path.isdir("neuralnet_info"):
      os.mkdir("neuralnet_info")


    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    s3.download_file(bucket_loc,
                  config_file_name,
                  MASK_NMF_CONFIG)
    s3.download_file(bucket_loc,
                  weights_file_name,
                  MASK_NMF_MODEL)



    model = maskrcnn_detector(MASK_NMF_MODEL,
                                MASK_NMF_CONFIG,
                                confidence,
                                allowed_overlap,
                                cpu_only, order=order)

    

    ##END OF PARAMETER DEFINITION



    if temporal_components is None:
      raise ValueError("Deconvolution was not run on this PMD output. Re-run PMD with deconv")


    # Run Detection On Select Frames
    print("Performing MaskRCNN detection...")
    bin_masks, footprints, properties, _ = segment_local_UV(
      U_sparse,
      mixing_weights,
      temporal_components,
      tuple((d1, d2, temporal_components.shape[-1])),
      model,
      frame_len,
      block_size=(block_dims_x, block_dims_y),
      order=order
    )

    print("Filtering detected components...")
    keep_masks = filter_components_UV(
          footprints, bin_masks, properties,
          spatial_thresholds[0], spatial_thresholds[1])
    print(f"Filtering completed {keep_masks.shape} of {np.count_nonzero(keep_masks)} components retained")

    a_dense = np.asarray(footprints[:, keep_masks].todense())
    # a_dense = np.asarray(footprints.todense())
    a = a_dense.reshape((d1, d2, -1), order=order)
    
    return a
  
        
@dash.callback(
    Output("secondpass_init", "children"), Output("placeholder_demix", "children"), Output("download_demixing_results", "data"), Output("placeholder_post_demixing", "children"), Input("button_id_demix", "n_clicks"), Input("button_id_demix_secondpass", "n_clicks"),
    prevent_initial_call=True
)
def demix_data(n_clicks, n_clickssecondpass):
    '''
    Contains algorithm for ROI detection via maskNMF/superpixels + localnmf demixing (or running superpixel + demixing)
    '''
    
    if not cache['PMD_flag']:
        return dash.no_update
    elif cache['PMD_object'] is None:
        return dash.no_update
    elif cache['first_pass_init_method'] == 'superpixel' and (cache['PMD_object'].a_init is None or cache['PMD_object'].c_init is None):
        return dash.no_update
    else:
        
        button_clicked = list(ctx.triggered_prop_ids.keys())[0]
        if button_clicked == "button_id_demix.n_clicks":
            print("FIRST DEMIX LOOP")

            if cache['first_pass_init_method'] == 'masknmf':
                print("Running masknmf dense init pipeline") 
                _ = dense_init_pipeline()
        else:
            print("SECOND DEMIX LOOP")
            
        
        print("results output folder before entering demix data is {}".format(cache['save_folder'])) 
        outdir = cache['save_folder']
        localnmf_params = cache['localnmf_params']


        length_cut=localnmf_params['length_cut']
        th=localnmf_params['length_cut']

        corr_th_fix=localnmf_params['corr_th_fix'] 
        switch_point = localnmf_params['switch_point']
        corr_th_fix_sec = localnmf_params['corr_th_fix_sec']
        corr_th_del = localnmf_params['corr_th_del']

        max_allow_neuron_size= localnmf_params['max_allow_neuron_size'] #0.15
        merge_corr_thr= localnmf_params['merge_corr_thr']
        merge_overlap_thr= localnmf_params['merge_overlap_thr']
        
        ###TODO: This should actually be used here instead of in the constructor for the pmd_object initialization. 
        ring_radius =  localnmf_params['r']
        pseudo_2 = localnmf_params['pseudo_2']



        
        residual_cut = localnmf_params['residual_cut']
        num_plane= localnmf_params['num_plane']
        patch_size= localnmf_params['patch_size'] 
        plot_en = localnmf_params['plot_en'] 
        text= localnmf_params['text']
        maxiter= localnmf_params['maxiter']
        update_after = localnmf_params['update_after']
        skips= localnmf_params['skips'] 
        plot_debug = False
        denoise = [False for i in range(maxiter)]
        for k in range(maxiter):
            if k > 0 and k % 8 == 0:
                denoise[k] = True

        
        my_pmd_object = cache['PMD_object']
            
        with torch.no_grad():
            
            a, c, b, X, W, res, corr_img_all_r, num_list = superpixel_analysis_ring.update_AC_bg_l2_Y_ring_lowrank(my_pmd_object, maxiter, corr_th_fix, corr_th_fix_sec, corr_th_del, switch_point, skips, merge_corr_thr, merge_overlap_thr, ring_radius, denoise=denoise, plot_en=plot_en, plot_debug=plot_debug, update_after=update_after)
            W_final = W.create_complete_ring_matrix(a)
            fin_rlt = {'U_sparse': my_pmd_object.U_sparse.cpu().to_scipy().tocsr(), 'R': my_pmd_object.R.cpu().numpy(), 'V': my_pmd_object.V.cpu().numpy(), 'a':a, 'c':c, 'b':b, "W":W_final, 'res':res, 'corr_img_all_r':corr_img_all_r, 'num_list':num_list, 'data_order': my_pmd_object.data_order, 'data_shape':my_pmd_object.shape};
            
        cache['PMD_object'] = my_pmd_object
            
        
        fin_rlt['data_shape'] = cache['shape']
        fin_rlt['data_order'] = cache['order']

        save_path = os.path.join(cache['save_folder'], "demixingresults.npz")
        np.savez(save_path, final_results = fin_rlt)
        cache['demixing_results'] = fin_rlt

        if button_clicked == "button_id_demix.n_clicks":
            return " ", dash.no_update, dcc.send_file(save_path), ""
        else:
            return dash.no_update, dash.no_update, dcc.send_file(save_path), ""


######
## Second pass plotting functionality
######

@app.callback(
    Output("local_correlation_plot_secondpass", "figure"), 
    Input("secondpass_init", "children"),
    prevent_initial_call=True
)
def get_residual_corr_img(value):
    if cache['PMD_object'] is not None:
        my_pmd_object = cache['PMD_object']
        if my_pmd_object.a is not None and my_pmd_object.c is not None:
            output = my_pmd_object.compute_local_correlation_image(0).reshape((my_pmd_object.shape[0], my_pmd_object.shape[1]), \
                                                                      order=my_pmd_object.data_order)
            my_fig = px.imshow(output)
            my_fig.update_coloraxes(showscale=False)
            my_fig.update_yaxes(showticklabels=False)
            my_fig.update_layout(title_text="Residual Corr. Image", title_x=0.5)
            return my_fig
        else:
            return dash.no_update
    else:
        return dash.no_update
    
@app.callback(
    Output("local_pixel_corr_plot_secondpass", "figure"),
    Input("local_correlation_plot_secondpass", "clickData"),
    Input("local_correlation_plot_secondpass", "figure"),
    prevent_initial_call=True
)
def update_residual_singlepixel_corr_plot(clickData, local_corr_fig):
    
    if cache['PMD_object'] is None:
        return dash.no_update
    elif cache['PMD_object'].a is None or cache['PMD_object'].c is None:
        return dash.no_update
    else:
        button_clicked = list(ctx.triggered_prop_ids.keys())[0]
        my_pmd_object = cache['PMD_object']
        if button_clicked == "local_correlation_plot_secondpass.clickData":
            x, y = get_points(clickData)
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

            temp_mat = np.arange(my_pmd_object.shape[0] * my_pmd_object.shape[1])
            temp_mat = temp_mat.reshape((my_pmd_object.shape[0], my_pmd_object.shape[1]), order=my_pmd_object.data_order)
            desired_index = temp_mat[y, x] ##Note y, x not x,y because the clickback returns the height as the second coordinate

            final_image = my_pmd_object.compute_single_pixel_correlation_image(desired_index, residual=True).cpu().numpy()
            final_image = final_image.reshape((my_pmd_object.shape[0], my_pmd_object.shape[1]), order=my_pmd_object.data_order)

            curr_fig = px.imshow(final_image, zmin=0, zmax=1)
            curr_fig.update_coloraxes(showscale=False)
            curr_fig.update(layout_coloraxis_showscale=False)
            curr_fig.update_yaxes(showticklabels=False)
            curr_fig.update_layout(title_text = "Residual Pixel Corr. Image at height = {}, width = {}".format(y,x),title_x=0.5)
            return curr_fig

        elif button_clicked== "local_correlation_plot_secondpass.figure":        

            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

            #Step 1: Get the local correlation figure
            my_img = np.array(local_corr_fig['data'][0]['z'])

            #Step 2: Get the indices of the "brightest" value
            x,y = np.unravel_index(my_img.argmax(), my_img.shape)

            
            #Step 3: Calculate the single pixel corr of this "brightest" value
            temp_mat = np.arange(cache['shape'][0] * cache['shape'][1])
            temp_mat = temp_mat.reshape((cache['shape'][0], cache['shape'][1]), order=cache['order'])
            desired_index = temp_mat[x, y] ##Note y, x not x,y because the clickback returns the height as the second coordinate


            final_image = my_pmd_object.compute_single_pixel_correlation_image(desired_index, residual=True).cpu().numpy()
            final_image = final_image.reshape((my_pmd_object.shape[0], my_pmd_object.shape[1]), order=my_pmd_object.data_order)

            curr_fig = px.imshow(final_image, zmin=0, zmax=1)
            curr_fig.update_yaxes(showticklabels=False)
            curr_fig.update_layout(title_text = "Pixel Corr. Image at ( {},{} )".format(y,x),title_x=0.5)
            curr_fig.update_coloraxes(showscale=False)
            curr_fig.update(layout_coloraxis_showscale=False)
            return curr_fig

        
@app.callback(
    Output("superpixel_plot_secondpass", "figure"),
    Input("superpixel_slider_secondpass", "value"),
    Input("local_correlation_plot_secondpass", "figure"),
    prevent_initial_call=True,
)
def generate_superpixel_plot_secondpass(value, fig):
    if cache['PMD_object'] is None or cache['PMD_object'].a is None or cache['PMD_object'].c is None:
        return dash.no_update
    else:
        my_pmd_object = cache['PMD_object']
        if torch.cuda.is_available():
            device='cuda'
        else:
            device = 'cpu'
        my_pmd_object = cache['PMD_object']
            
        lnmf_params = cache['localnmf_params']
        
        length_cut=lnmf_params['length_cut'][1] 
        th= lnmf_params['th'][1] 

        ##Do not need to modify
        residual_cut = lnmf_params['residual_cut'][1]
        num_plane=lnmf_params['num_plane']
        patch_size= lnmf_params['patch_size']
        plot_en = True
        text=True
        pseudo_2 = lnmf_params['pseudo_2'][1]


        cut_off_point = value
        my_pmd_object = cache['PMD_object']
        my_pmd_object.initialize_signals_superpixels(num_plane, cut_off_point, residual_cut, length_cut, th, pseudo_2, \
                                       text =True, plot_en = True)
        superpixel_image = my_pmd_object.superpixel_image_recent
        
        cache['PMD_object'] = my_pmd_object
        curr_fig = px.imshow(superpixel_image)
        curr_fig.update_yaxes(showticklabels=False)
        curr_fig.update_layout(title_text = "Superpixels Image, threshold = {}".format(value),title_x=0.5)
        
        ##Update the value
        lnmf_params = cache['localnmf_params']
        lnmf_params['superpixels_corr_thr'][1] = value
        cache['localnmf_params'] = lnmf_params
        return curr_fig
            
        
            
        
        
if __name__ == '__main__':
    port_number = 8900
    app.run_server(host='0.0.0.0',debug=True, port=port_number)