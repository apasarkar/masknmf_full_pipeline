# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import os
import dash
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State

import plotly.express as px
import pandas as pd

import time
import datetime
from dash import DiskcacheManager, CeleryManager, Input, Output, html
import shutil

### PRODUCTION VS NON PRODUCTION WORKER MANAGEMENT 
if 'REDIS_URL' in os.environ:
    # Use Redis & Celery if REDIS_URL set as an env variable
    from celery import Celery
    celery_app = Celery(__name__, broker=os.environ['REDIS_URL'], backend=os.environ['REDIS_URL'])
    background_callback_manager = CeleryManager(celery_app)

else:
    # Diskcache for non-production apps when developing locally
    import diskcache
    cache = diskcache.Cache("./cache")
    background_callback_manager = DiskcacheManager(cache)



def list_dir(path):
    print("input to list_dir is {}".format(path))
    return os.listdir(path) + [".."]



app = Dash(__name__, background_callback_manager=background_callback_manager)

##Globally used data structures/info
'''
Structure of this app: we define the data structures used in the algorithm (for e.g. the dictionary containing all parameters)
as global variables, which can be modified by the callback functions
'''
no_file_flag = ""
present_dir = [os.getcwd(), no_file_flag] #Global variable which will be modified during file selection
results_output_folder = [""]

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
    'block_height':32,
    'block_width':32,
    'overlaps_height':10,
    'overlaps_width':10,
    'window_length':6000,
    'background_rank':15,
    'deconvolve':True,

}

### End of globally used data structures

controls = [
    dcc.Dropdown(
        id="dropdown",
        options=[{"label": x, "value": x} for x in list_dir(present_dir[0])],
        value="",
    )
]

app.layout = html.Div(
    # [html.H1("File Selected: None"), html.Div(controls), html.Div(id="folder-files")]
    [html.H1("Please select a multipage tiff file using the dropdown below"),\
     html.H1("File Selected: None",id="folder-files"),\
     html.H1("Current Folder: {}".format(present_dir), id="curr_folder"),\
     html.Div(controls), \
    ### Motion Correction Layout## 
     html.H1("Step 1: Motion Correction + PMD compression and denoising. Specify paramters and hit SUBMIT to run"),\
     html.Div(
            [
                html.Div(id='placeholder', children=""),
                # html.Progress(id="progress_bar", value="0"),
            ]
        ),\
     html.Button(id="button_id", children="Run Job!"),\
     ### Demixing ### 
     html.H1("Step 2: Demixing. Specify paramters and hit SUBMIT to run"),\
     html.Div(
            [
                html.Div(id='placeholder_demix', children=""),
                # html.Progress(id="progress_bar", value="0"),
            ]
        ),\
     html.Button(id="button_id_demix", children="Run Job!"),\

    ]
)

### CALLBACKS FOR FILE INPUT

@app.callback(Output("dropdown", "value"), Output("folder-files", "children"), Output("curr_folder", "children"), Output("dropdown", "options"), Input("dropdown", "value"))
def list_all_files(folder_name):

    #Decide if input is a file or not:
    
    folder_response = "Current Folder {}"
    default_value = ""
    
    decided_path = os.path.normpath(os.path.join(present_dir[0], folder_name))
    is_file = os.path.isfile(decided_path)
    is_dir = os.path.isdir(decided_path)

    if is_file:
        #A file has been selected, need to update texts: 
        new_text = "File Selected: {}".format(folder_name)
        current_folder = present_dir[0]
        present_dir[1] = decided_path
        final_dir = [{"label": x, "value": x} for x in list_dir(current_folder)]
        return default_value, new_text, folder_response.format(current_folder), final_dir
    elif is_dir:
        present_dir[1] = no_file_flag
        present_dir[0] = decided_path
        final_dir = [{"label": x, "value": x} for x in list_dir(decided_path)]
        return default_value, "File Selected: None", folder_response.format(present_dir[0]), final_dir
    else:
        raise ValueError("Invalid suggestion")

## MOTION CORRECTION + PMD COMPRESSION CALLBACKS


@dash.callback(
    output=Output("placeholder", "children"),
    inputs=Input("button_id", "n_clicks"),
    background=True,
    running=[
        (Output("button_id", "disabled"), True, False),
        (
            Output("placeholder", "style"),
            {"visibility": "hidden"},
            {"visibility": "visible"},
        ),
    ],
    # progress=[Output("progress_bar", "value"), Output("progress_bar", "max")],
    prevent_initial_call=True
)
def register_and_compress_data(n_clicks):
    data_folder = present_dir[0]
    input_file = present_dir[1]

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
        print("creating folder at location {} in gdrive".format(final_path))
        return final_path


    #NOTE: this data folder will also contain the location of the TestData
    data_folder = set_and_create_folder_path(present_dir[1], present_dir[0])
    results_output_folder[0] = data_folder
    input_file = present_dir[1]


    register = mc_params['register']#True #@param {type:"boolean"}
    # devel = True #@param {type:"boolean"}
    devel = mc_params['devel']#True #never delete uploaded data

    dx = mc_params['dx'] #2 #@param {type:"slider", min:0, max:100, step:1}
    dy = mc_params['dy'] #2 #@param {type:"slider", min:0, max:100, step:1}

    dxy = (dx, dy)

    max_shift_in_um_xdimension = mc_params['max_shift_in_um'][0]#50 #@param {type:"slider", min:0, max:200, step:1}
    max_shift_in_um_ydimension = mc_params['max_shift_in_um'][1]#50 #@param {type:"slider", min:0, max:200, step:1}

    max_shift_um = (max_shift_in_um_xdimension, max_shift_in_um_ydimension)

    max_deviation_rigid = mc_params['max_deviation_rigid'] #5 #@param {type:"slider", min:0, max:100, step:1}

    patch_motion_um_x = mc_params['patch_motion_um'][0] #17 #@param {type:"slider", min:0, max:200, step:1}
    patch_motion_um_y = mc_params['patch_motion_um'][1] #17 #@param {type:"slider", min:0, max:200, step:1}

    patch_motion_um = (patch_motion_um_x, patch_motion_um_y)

    overlaps_x = mc_params['overlaps'][0] #24 #@param {type:"slider", min:0, max:200, step:1}
    overlaps_y = mc_params['overlaps'][1] #24 #@param {type:"slider", min:0, max:200, step:1}

    overlaps = (overlaps_x, overlaps_y)

    border_nan = 'copy'

    niter_rig = mc_params['niter_rig'] #2 #@param {type:"slider", min:1, max:10, step:1}
    niter_els = mc_params['niter_els'] #2 #@param {type:"slider", min:1, max:10, step:1}

    pw_rigid = mc_params['pw_rigid'] #True #@param {type:"boolean"}

    use_gSig_filt = mc_params['use_gSig_filt'] #False #@param {type:"boolean"}
    gSig_filt_x = mc_params['gSig_filt'] #3 #@param {type:"slider", min:0, max:30, step:1}
    gSig_filt_y = mc_params['gSig_filt'] #3 #@param {type:"slider", min:0, max:30, step:1}
    
    sketch_template = True

    if use_gSig_filt: 
        gSig_filt = (gSig_filt_x, gSig_filt_y)
    else:
        gSig_filt = None

    frames_per_split =  500
    # frames_per_split = 200

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

        print("the value of niter_rig is {}".format(niter_rig))

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
        'min_mov': None,                    # minimum value of movie
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

        print("THE MC DICT IS {}".format(mc_dict))


        corrector = motion_correction.MotionCorrect(target, dview=dview, **mc_dict)

        # Run MC, Always Saving Non-Final Outputs For Use In Next Iteration
        corrector_obj = corrector.motion_correct(
            save_movie=False
        )


        # target = corrector.target_file


        display("Motion correction completed.")


        # Save Frame-wise Shifts
        display(f"Saving computed shifts to ({outdir})...")
        np.savez(os.path.join(outdir, "shifts.npz"),
                 shifts_rig=corrector.shifts_rig,
                 x_shifts_els=corrector.x_shifts_els if pw_rigid else None,
                 y_shifts_els=corrector.y_shifts_els if pw_rigid else None)
        display('Shifts saved as "shifts.npz".')


        return corrector_obj, target


    # Get data and config filenames from cmd line args
    data_name = input_file
    outdir = results_output_folder[0]
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    # Load User-Provided Params
    params = load_config(INPUT_PARAMS)
    print(params)
    write_params(os.path.join(outdir, "MotionCorrectConfig.yaml"),
                  default=INPUT_PARAMS,
                  **params)


    # %load_ext line_profiler
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
    data_folder = results_output_folder[0]

    block_height = pmd_params['block_height'] #32 #@param {type:"slider", min:20, max:100, step:4}
    block_width = pmd_params['block_width'] #32 #@param {type:"slider", min:20, max:100, step:4}
    block_sizes = [block_height, block_width]

    overlaps_height = pmd_params['overlaps_height'] #10 #@param {type:"slider", min:0, max:100, step:1}
    overlaps_width = pmd_params['overlaps_width'] #10 #@param {type:"slider", min:0, max:100, step:1}

    if overlaps_height > block_height: 
        print("Overlaps height was set to be greater than block height, which is not valid")
        print("Setting overlaps to be 5")
        overlaps_height = 5

    if overlaps_width > block_width:
        print("Overlaps width was set to be greater than width height, which is not valid \
        Setting overlaps to be 5")
        overlaps_width = 5

    overlap = [overlaps_height, overlaps_width]


    window_length = pmd_params['window_length'] #6000 #@param {type:"integer"}
    if window_length <= 0:
        print("Window length cannot be negative! Resetting to 6000")
        window_length = 6000
    start = 0
    end = window_length

    # background_rank = 0
    background_rank = pmd_params['background_rank'] #15 #@param {type:"slider", min:0, max:100, step:1}

    # rank_prune_factor = 0.25 #@param {type:'slider', min:0, max:1, step:0.01}

    deconvolve=True
    deconv_batch=1000
    # deconvolve=True #@param {type:'boolean'}
    # deconv_batch=1000  #@param {type:'slider', min:1000, max:30000, step:1000}

    ###THESE PARAMS ARE NOT MODIFIED
    # num_sims = 64
    sim_conf = 5
    max_rank_per_block = 40 #@param {type:"slider", min:5, max:50, step:1}

    #@markdown Keep run_deconv true unless you do not want to run maskNMF demixing
    run_deconv = True #@param {type:'boolean'}
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


    ## PMD: Run the matrix decomposition pipeline 

    ##TODO: Add apriori SVD option
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
        from masknmf.engine.sparsify import get_factorized_projection
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

        start, end = frame_range[0], frame_range[1]


        #Reshape U using order="F" here
        U, R, s, V, tiff_loader_obj = localmd_decomposition(input_file, block_sizes, overlap, [start, end], \
                                        max_components=max_components, background_rank = background_rank, sim_conf=sim_conf,\
                                         tiff_batch_size=tiff_batch_size,pixel_batch_size=pixel_batch_size, batching=batching, dtype=dtype, order=order, \
                                         num_workers=0, frame_corrector_obj = corrector)


        ## Step 2h: Run deconvolution:
        limit = 5000
        if run_deconv:
            np.savez("Deconvolution_Testing.npz", U=U, R=R, s=s, V=V, batch_size=deconv_batch, allow_pickle=True)
            deconv_components = get_factorized_projection(
              U,
              R,
              s[:, None] * V[:, :limit],
              batch_size=deconv_batch
          )
        else:
            display("WARNING: YOU ARE NOT USING THE DECONVOLUTION STEP, MASKNMF WILL NOT PERFORM AS WELL.")
            deconv_components = s[:, None] * V[:, :limit]


        def save_decomposition(U, R, s, V, deconvolved_temporal, load_obj, folder, order="F"):
            file_name = "decomposition.npz"
            final_path = os.path.join(folder, file_name)

            np.savez(final_path, fov_shape = load_obj.shape[:2], \
                fov_order=order, U_data = U.data, \
                U_indices = U.indices,\
                U_indptr=U.indptr, \
                U_shape = U.shape, \
                U_format = type(U), \
                R = R, \
                s = s, \
                Vt = V, \
                deconvolved_temporal=deconvolved_temporal, \
                 mean_img = tiff_loader_obj.mean_img, \
                 noise_var_img = tiff_loader_obj.std_img)

            display("the decomposition.npz file is saved at {}".format(folder))


        ##Step 2i: Save the results: 
        save_decomposition(U.tocsr(), R, s, V, deconv_components, tiff_loader_obj, folder, order=order)



    tiff_batch_size = 1000
    try:
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
    except FileNotFoundError:
        print("\n \n \n")
        display("--------ERROR GENERATED, DETAILS BELOW-----")
        display("The file was not located. Please consider specifying the file again (step 0) and run the entire pipeline from the start\
        (Motion Correction)")
        import jax
        import torch
        jax.clear_backends()
        torch.cuda.empty_cache()
        display("Cleared memory")
    except Exception as e:
        print("\n \n \n")
        display("--------ERROR GENERATED, DETAILS BELOW-----")
        display("Unexpected error, please report")
        import jax
        import torch
        jax.clear_backends()
        torch.cuda.empty_cache()
        display("Cleared backends")
        print(e)
        display("Please re-run the pipeline starting from motion correction.")


@dash.callback(
    output=Output("placeholder_demix", "children"),
    inputs=Input("button_id_demix", "n_clicks"),
    background=True,
    running=[
        (Output("button_id_demix", "disabled"), True, False),
        (
            Output("placeholder_demix", "style"),
            {"visibility": "hidden"},
            {"visibility": "visible"},
        ),
    ],
    prevent_initial_call=True
)
def demix_data():
    '''
    Contains algorithm for maskNMF detection + demixing (or running superpixel + demixing)
    '''
    
    demix_params = {
        
        
        
    }

#NOTE: this data folder will also contain the location of the TestData
data_folder = results_output_folder[0]

## @markdown #Step 2 MaskNMF: Set parameter values for initializing the neural network (see documentation below)

#Specify related parameters: 
confidence = 0.5 #@param {type:"slider", min:0, max:1, step:0.01}
allowed_overlap = 70 #@param {type:"slider", min:0, max:300, step:10}
cpu_only = False #Whether to run net on CPU (very slow) or GPU
# order = order #The default ordering of PMD outputs, specified in a previous block




## @markdown #Step 3 MaskNMF: Specify the key parameter values for Mask R-CNN detection of neural signals (see documentation above)

##PARAMETERS
block_dims_x = 20 #@param {type:"slider", min:5, max:200, step:1}
block_dims_y = 20 #@param {type:"slider", min:5, max:200, step:1}
frame_len = 200 #@param {type:"slider", min:5, max:2000, step:5}
#In each local spatial patch, we look at the "frame_len"-th brightest frames

'''When we filter our large list of neurons, we use these 
thresholds to discard similar neurons (to avoid over initializing the same cell)
'''
spatial_thresholds_1 = 0.3 #@param {type:"slider", min:0, max:1, step:0.01}
spatial_thresholds_2 = 0.3 #@param {type:"slider", min:0, max:1, step:0.01}

spatial_thresholds = [spatial_thresholds_1, spatial_thresholds_2]



def run_masknmf(data_folder, input_file, confidence, allowed_overlap, cpu_only,\
                block_dims_x, block_dims_y, frame_len, spatial_thresholds):

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

  #Ring-LocalNMF specific imports
  from localnmf import superpixel_analysis_ring


  import boto3
  from botocore.config import Config
  from botocore import UNSIGNED

  from masknmf.engine.segmentation import segment_local_UV, filter_components_UV
  from masknmf.detection.maskrcnn_detector import maskrcnn_detector

  input_file = os.path.join(data_folder, "decomposition.npz")

  data = np.load(input_file, allow_pickle=True)

  U_sparse = scipy.sparse.csr_matrix(
          (data['U_data'], data['U_indices'], data['U_indptr']),
          shape=data['U_shape']
      ).tocoo()

  order = data.get('fov_order', np.array("C")).item()
  U = np.array(U_sparse.todense())
  shape = data['U_shape']
  d1,d2 = data['fov_shape']
  R = data['R']
  s = data['s']
  Vt = data['Vt']
  T = Vt.shape[1]
  V_full = R.dot(s[:, None] *Vt)
  V = V_full
  dims = (d1, d2, T)
  U_r = U.reshape((d1, d2,-1), order=order)


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

  temporal_components = data.get('deconvolved_temporal')
  mixing_weights = data['R']

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
  a = a_dense.reshape((d1, d2, -1), order=order)

  return a

try: 

  import torch
  import jax
  torch.cuda.empty_cache()
  jax.clear_backends()


  a = run_masknmf(data_folder, input_file, confidence, allowed_overlap, cpu_only,\
                block_dims_x, block_dims_y, frame_len, spatial_thresholds)
  


  torch.cuda.empty_cache()
  jax.clear_backends()

except KeyboardException:
  display("\n \n \n")
  display("-------- ERROR GENERATED----------")
  display("The user manually ended the program execution. Please re-run this code block.")
  import torch
  import jax
  torch.cuda.empty_cache()
  jax.clear_backends()
  display("Memory Cleared, Ready to Re-Run")
except:
  display("\n \n \n")
  display("-------- ERROR GENERATED----------")
  display("Miscellaneous Error, please try to re-run.")
  import torch
  import jax
  torch.cuda.empty_cache()
  jax.clear_backends()
  display("Memory Cleared, Ready to Re-Run")




if __name__ == '__main__':
    app.run_server(debug=True)

