# SESimulate

SE Simulate seeks to simulate the self-efficacy of given networks of 
populations, given initial parameters of the coaching effectiveness,
number, and social network clustering. Of interest were the time
effects of SE development both in individuals and the entire network
as a whole, inspiring the outputs provided. Given these parameters,
thus, the following graphical outputs are provided:

- Network depictions (20 time step intervals): Portrays individuals
(red denoting those lacking coaches and blue for those with them)
with their transparency further corresponding to their SE level
- Graph of the individual changes in exercise and SE levels
- Graph depicting the relation between SE and exercise level as a 
result of changing the following parameters;
	- Coaching effectiveness
	- Coach count
	- Time decay constants
	- Different network clustering algorithsm

# Results

The results of SE Simulate are outputted in the Results folder 
(originally blank) and are further sorted by the time results (graphical 
displays for steps of time) and sensitivity (for sensitivity analysis).

Note: The package uses Python 3, with the Numpy, NetworkX, and
Matplotlib libraries.