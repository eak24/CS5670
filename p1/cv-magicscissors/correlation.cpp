#include "correlation.h"

/************************ TODO 2 **************************/
/*
 *	INPUT:
 *		origImg:		the original image,
 *		imgWidth:		the width of the image
 *		imgHeight:		the height of the image
 *						the image is arranged such that
 *						origImg[3*(row*imgWidth+column)+0],
 *						origImg[3*(row*imgWidth+column)+1],
 *						origImg[3*(row*imgWidth+column)+2]
 *						are R, G, B values for pixel at (column, row).
 *
 *      kernel:			the 2D filter kernel,
 *		knlWidth:		the width of the kernel
 *		knlHeight:		the height of the kernel
 *
 *		scale, offset:  after correlating the kernel with the origImg,
 *						each pixel should be divided by scale and then added by offset
 *
 *		selection:      a byte array of the same size as the image,
 *						indicating where in the original image should be filtered, e.g.,
 *						selection[k] == 1 ==> pixel k should be filtered
 *                      selection[k] == 0 ==> pixel k should NOT be filtered
 *                      a special case is selection is a NULL pointer, which means all the pixels should be filtered.
 *
 *  OUTPUT:
 *		rsltImg:		the filtered image of the same size as original image.
 *						it is a valid pointer ( allocated already ).
 */

void image_filter(double* rsltImg, const unsigned char* origImg, const unsigned char* selection,
                  int imgWidth, int imgHeight,
                  const double* kernel, int knlWidth, int knlHeight,
                  double scale, double offset)
{
    // Note: copying origImg to rsltImg is NOT the solution, it does nothing!
    int i,j,p;
    double rsltPixel[3];
    p=0;
    for (i=0;i<imgHeight;i++) //row
    	for(j=0;j<imgWidth;i++) //col
    	{
    		if (selection && selection[p]){ //only work on pixels for which selection[pixel]=1
    			pixel_filter(&rsltImg[3*p],
    				j,
    				i,
    				origImg,
    				imgWidth,
    				imgHeight,
    				kernel,
    				knlWidth,
    				knlHeight,
    				scale,
    				offset);
    		}
    		p+=1;
    	}
}

/************************ END OF TODO 2 **************************/


/************************ TODO 3 **************************/
/*
 *	INPUT:
 *      x:				a column index,
 *      y:				a row index,
 *		origImg:		the original image,
 *		imgWidth:		the width of the image
 *		imgHeight:		the height of the image
 *						the image is arranged such that
 *						origImg[3*(row*imgWidth+column)+0],
 *						origImg[3*(row*imgWidth+column)+1],
 *						origImg[3*(row*imgWidth+column)+2]
 *						are R, G, B values for pixel at (column, row).
 *
 *      kernel:			the 2D filter kernel,
 *		knlWidth:		the width of the kernel
 *		knlHeight:		the height of the kernel
 *
 *		scale, offset:  after correlating the kernel with the origImg,
 *						the result pixel should be divided by scale and then added by offset
 *
 *  OUTPUT:
 *		rsltPixel[0], rsltPixel[1], rsltPixel[2]:
 *						the filtered pixel R, G, B values at row y , column x;
 */
#define ABS(x)           (((x) < 0) ? -(x) : (x))

void pixel_filter(double rsltPixel[3], int x, int y, const unsigned char* origImg, int imgWidth, int imgHeight,
                  const double* kernel, int knlWidth, int knlHeight,
                  double scale, double offset)
{
	int i,j,imx,imy;
	double kw;
	const unsigned char *  imgPtr;
	rsltPixel[0]=0.0;rsltPixel[1]=0.0;rsltPixel[2]=0.0; 
	//Clear the pixels, just in case;

	for (i=0;i<knlHeight;i++)
		for(j=0;j<knlWidth;j++){
			imx = ABS((knlWidth/2)-j+x); imy=ABS((knlHeight/2) - i+ y); //If we go negative, reflect around 0.
			if (imx>=imgWidth)
				imx = imgWidth - (imx-imgWidth);
				//If we go past the width, reflect around
			if (imy>=imgHeight)
				imy = imgHeight - (imy-imgHeight);
				//If we go past the height, reflefct around
			kw = kernel[i*knlWidth+j]; //The lcoation in the kernel
			imgPtr = & origImg[3*(imy * imgWidth + imx)]; //The location of the 3 rgb bytes
			rsltPixel[0]+= ((double) imgPtr[0])*kw;
			rsltPixel[1]+= ((double) imgPtr[1])*kw;
			rsltPixel[2]+= ((double) imgPtr[2])*kw;
					}
			rsltPixel[0]/=scale;rsltPixel[1]/=scale;rsltPixel[2]/=scale;
			rsltPixel[0]+=offset;rsltPixel[1]+=offset;rsltPixel[2]+=offset;

}

/************************ END OF TODO 3 **************************/

