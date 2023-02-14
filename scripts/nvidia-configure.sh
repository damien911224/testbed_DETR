sudo service lightdm start

DISPLAY=:0 XAUTHORITY=/var/run/lightdm/root/:0 sudo nvidia-settings -a [gpu:0]/GPUFanControlState=1
DISPLAY=:0 XAUTHORITY=/var/run/lightdm/root/:0 sudo nvidia-settings -a [fan:0]/GPUTargetFanSpeed=80

DISPLAY=:0 XAUTHORITY=/var/run/lightdm/root/:0 sudo nvidia-settings -a [gpu:1]/GPUFanControlState=1
DISPLAY=:0 XAUTHORITY=/var/run/lightdm/root/:0 sudo nvidia-settings -a [fan:1]/GPUTargetFanSpeed=80

# sudo nvidia-smi -pl 220
# sudo nvidia-smi -pm 1

# sudo service lightdm start
