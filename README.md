DQN-Hind-Sight-Reward-Shaping
===============
Hind Sight Reward Shaping is a simple and effective reward shaping technique that exploits the fundamentals of the Bellman equation and experience replay in current state-of the-art Deep Reinforcement Learning algorithms like DQN.

[![Image](./image.JPG)](https://github.com/ByronDev121/DQN-Hind-Sight-Reward-Shaping/)

Building AirSim
--------------
This project uses Microsoft's AirSim simulator for drones, cars and more, built on Unreal Engine. Follow the build instructions [here](https://microsoft.github.io/AirSim/).

Creating your environment
--------------
To install all dependecies required for this project there is a requirements.txt file included in the repository. Simply create a new enviroment in anaconda or what ever container manager you are using and run the following command:

```bash
pip install -r requirements.txt 
```

Creating your own track in Unreal Engine
--------------
To create your own track I suggest following [this](https://www.youtube.com/watch?v=wR0fH6O9jD8) tutorial. When you have built your track, add it into your AirSim project and build from source. 

I found these quite helpfull:

https://microsoft.github.io/AirSim/docs/unreal_custenv/

https://www.youtube.com/watch?v=1oY8Qu5maQQ&feature=youtu.be

Usage
--------------
## Open Track in AirSim
Open the airsim application with you track. e.g. (Open the working directory of your AirSim executable on a windows machine)
```bash
AirSim.exe /Game/OvalTrack -windowed
```
## Train Model
From the project's directory execute training code in your Anaconda environment
```bash
~(DQN-HRS)>python3 main.py
```

## Run Model
From the project's directory execute training code in your Anaconda environment (remember)
```bash
~(DQN-HRS)>python3 run_agent.py
```

Conference Paper
------
This research has been publish under IEEE. Find the paper here. 

Video Results
------
The video result can be found [here](https://www.youtube.com/watch?v=dJN05nHdvpE&t=311s)

License
-------

The code in this repository is distributed under the MIT License.


