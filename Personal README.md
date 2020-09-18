
Personal README:
In order to successfully and cleanly launch Open Pose, one has to do two steps:
1. Set up a working environment to run Open Pose (I did it with Anaconda):
	First, an install Anaconda, as it's ability to run multiple separate environments is useful;
	Run the Anaconda Prompt (preferably as Administrator to avoid permission issues);
	Check if everything is working (typing 'conda' should bring up a list of commands);
	Create an environment (conda create -n <insert env_name here>);
	Activate the environment (conda activate <env_name>);

	Next, the environment needed to be filled with the appropriate tools (the env_name should be seen next to file path);
	Install python (I used Python 3.7.6 for this, so 'conda install python=3.7.6');
	Install tensorflow ('conda install tensorflow'. WARNING: If TensorFlow 2 is too advanced for the code, use 'conda install tensorflow=1.14');
	Install OpenCV ('conda install opencv'. If that does not work, 'conda install opencv-python');
	Clone the Open Pose Git repository to the machine (For that, use the Prompt to 'cd <path>' into the desired folder, and execute 'git clone https://www.github.com/ildoonet/tf-pose-estimation' ); 
	Execute setup.py (For that, go to the cloned repository with 'cd <path>/tf-pose-estimation' and execute the setup with 'python setup.py install');
	Install the Requirements (While still being in the tf-pose-estimation folder, run 'conda install -r requirements.txt'. This will automatically install all the necessary libraries and dependencies. )
	
2. Once the environment is set up with all the necessary tools, additional steps are needed specifically to set up Open Pose;
	Install Swig ('conda install swig');
	Build a library for C++ post processing (Run 'swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace'. If that doesn't work, cd to the location of pafprocess.i and run the code from there. Also try replacing 'python3' for 'python')
	WARNING: For this step, the machine needs to have the ability to run BASH scripts. If it doesn't, download a BASH running tool (I used Git. "https://git-scm.com/downloads")
	With your BASH running terminal, activate the conda environment, and go to tf-pose-estimation/models/graph/cmu (cd <path>/tf-pose-estimation/models/graph/cmu)
	While inside /tf-pose-estimation/models/graph/cmu, run the download shell. ('bash download.sh') [WARNING: You need a BASH running tool of this, if it is done on Windows.]
	
3. Once everything is done and set up, only launching and testing remains;
	Run a test for Pose Detection in images ('python run.py --model=mobilenet_thin --image=./images/p1.jpg')  [Note: The images are located in /tf-pose-estimation/images]
	Run a test for Pose Detection in video ('python run_video.py --model=mobilenet_thin --video=./images/<insert video name and type>')
	Run a test for Pose Detection in a webcam ('python run_webcam.py --model=mobilenet_thin --camera=0')
	Run a test for Pose Detection in a webcam with TensoRT ('python run_webcam.py --model=mobilenet_thin --camera=0 --tensorrt=True')