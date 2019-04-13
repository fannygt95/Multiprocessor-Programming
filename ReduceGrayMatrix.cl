__constant sampler_ttmp = CLK_NORMALIZED_COORDS_FALSE| CLK_ADDRESS_CLAMP_TO_EDGE| CLK_FILTER_NEAREST;
	
__kernel void ReduceGrayMatrix(__read_only image2d_t image0, __read_only image2d_t image1, __global uchar *reduceim0, __global  uchar *reduceim1, int width, int height) {
		
    constinti = get_global_id(0);
		constint j = get_global_id(1);

	  int2 Index = { (4*j - 1*(j >0)), (4*i - 1*(i>0)) };
    
    uint4 pixel0  = read_imageui(image0, tmp, Index);
	  uint4 pixel1 = read_imageui(image1, tmp, Index);
	
	reduceim0[i*width+j] = 0.2126*pixel0.x  +0.7152*pixel0.y  + 0.0722*pixel0.z;
	reduceim1[i*width+j] = 0.2126*pixel1.x + 0.7152*pixel1.y + 0.0722*pixel1.z;
	}
