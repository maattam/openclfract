//
//  Author  : Matti Määttä
//  Summary : OpenCL kernel for calculating mandelbrot depth at screen-space coordinate.
//

#if USE_DOUBLE

	#pragma OPENCL EXTENSION cl_khr_fp64 : enable

	typedef double real_t;
	typedef double2 real_t2;

#else

	typedef float real_t;
	typedef float2 real_t2;

#endif

bool cardoid_check(real_t2 c)
{
	real_t dy = c.y*c.y;
	real_t q = (c.x - 0.25)*(c.x - 0.25) + dy;
	if(q * (q + c.x - 0.25) < 0.25 * dy)
		return true;

	return false;
}

int calculate_depth(real_t2 c, int max_iters)
{
	// Check if point lies within the cardioid
	if(cardoid_check(c))
		return max_iters;

	int iters = 0;
	real_t2 z1 = 0;
	real_t2 z2 = 0;

	while(z2.x + z2.y < 4 && iters < max_iters)
	{
		// Z = Z^2 + C
		z1.y = z1.y * z1.x;
		z1.y += z1.y;
		z1.y += c.y;
		z1.x = z2.x - z2.y + c.x;
		z2 = z1*z1;

		iters++;
	}
	
	return iters;
}

kernel void mandelbrot(write_only image2d_t out, int width, int height,
		global real_t* view, uint max_iterations, global float4* color_map)
{
	const real_t2 fact = (real_t2)((view[1] - view[0]) / width, (view[3] - view[2]) / height);
	const int2 coord = (int2)(get_global_id(0), get_global_id(1));

	// Map current texture coordinate as mandelbrot coordinate
	real_t2 ref = (real_t2)(view[0] + coord.x*fact.x, view[3] - coord.y*fact.y);

	int depth = calculate_depth(ref, max_iterations);
	write_imagef(out, coord, color_map[depth]);
}