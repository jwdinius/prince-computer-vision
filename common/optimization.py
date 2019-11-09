def direct_line_search_1d(cost_fn, bounds, args, kwargs):
	# See Appendix B.3: Line Search
	# FYI - this routine is pretty slow...
	a, d = bounds
	try:
		stopping = kwargs["stopping_thresh"]
	except KeyError:
		stopping = 1.0
	its=0
	while True:
		its+=1
		third = (d - a) / 3.
		b = a + third
		c = d - third
		b_cost = cost_fn(b, *args)
		c_cost = cost_fn(c, *args)
		if b_cost < c_cost:
			d = c  # new search window is [a, c]
		else:
			a = b  # new search window is [b, d]

		if d - a < stopping:
			return d  # could also return a