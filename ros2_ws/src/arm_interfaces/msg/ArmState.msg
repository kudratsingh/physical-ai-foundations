# Measured state. Header uses system ROS time for transport-age measurements;
# sim_time is the separate physics clock. Reset increments epoch.
std_msgs/Header header
uint32 epoch
uint64 sequence
float64 sim_time
float64[2] position
float64[2] velocity
float64[2] effort
bool paused
