# One immediate joint command, correlated to the state used to compute it.
std_msgs/Header header
builtin_interfaces/Time observation_stamp
uint32 epoch
uint64 observation_sequence
float64[2] position
float64[2] velocity
float64[2] torque
