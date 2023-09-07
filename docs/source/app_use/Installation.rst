.. maskNMF app installation guide. 

Installation
============

Containerized Deployment
------------------------

The app is provided via a docker container for ease of use and reproducibility. These are the basic steps to run the app on a GPU-enabled linux computer. We address two separate common use cases here. In the "remote access" case, you access the app on the server from, say, a personal computer. This is done via an ssh connection. Our instructions in this case assume you are using a linux/mac personal computer. In the "local access" case, you are sitting at the linux computer and can directly access the app from the computer's monitor. Note that the remote access case also outlines the simple steps needed to deploy this pipeline on the cloud (via AWS for example). 

Step 0: There are some basic dependencies you will need on your linux machine: 

.. bulletlist:: Links
   * make (most Linux operating systems come with this, to check if you have it, run "make --version" on the terminal. Otherwise you will need to install it 
   * docker
   * nvidia-docker
   * git
   * GPU drivers supporting CUDA 11.* applications 
   
If using AWS, we recommend you launch from a linux AMI that (a) supports GPU use and (b) already has these basic dependencies. A great example is the NVIDIA GPU-Optimized AMI.
Step 1: Clone the repo. Get the masknmf app repository onto the computer: 

.. code-block:: bash
    cd
    git clone https://github.com/apasarkar/masknmf_full_pipeline.git

Step 2: Upload the relevant dataset to the computer.

If you have local access, you can do this step physically (for example, by using a hard drive to upload the data onto the system). 

If you have remote access, you can do the following on the command line: 

.. code-block:: bash
    scp <Local_Path_To_Dataset> <Server_IP_Address>:<Some_Location_On_Server>

Step 3: Launch the app on the remote instance
You need to do the following: 

.. code-block:: bash
    cd
    cd masknmf_full_pipeline
    make launch dataname=<Absolute_Path_To_Your_Dataset>

Step 4: Connect your local computer to the server to view/control the app.

If you have local access, you can skip this step entirely. 

If you have remote access, note that he maskNMF app is currently exposed via port 8981 on the linux server whenever it is launched. So you need to set up port forwarding from your personal computer to port 8981 on the linux server. To do this, run the following on the linux command line: 

.. code-block:: bash
    ssh -N -f -L localhost:2003:localhost:8981 <Server IP Address>

where <Server IP Address> is the IP address of the server.

Step 5: Access the app

If you have local access to the computer, navigate to a browser and type "localhost:8981" to access the app. 

If you have remote access, navigate to your browser and type "localhost:<X>" where <X> is the port on your computer which you connected to the remote linux computer's port 8981. Using the example from step 3, you might enter "localhost:2003" in your browser. 


Local Deployment without container
----------------------------------
It is also possible to run this pipeline without docker, though this is not recommended and will require careful installation of GPU compute libraries. See the dockerfile in the repository for a starting point which can be modified for your system.




