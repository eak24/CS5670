/***************************************************************
 * CS4670/5670, Fall 2012 Project 4
 * File to be modified #2:
 * ImgView.inl (included from ImgView.cpp)
 *		contains routines for computing the 3D position of points
 ***************************************************************/

//
// TODO 4: sameXY()
//		Computes the 3D position of newPoint using knownPoint
//		that has the same X and Y coordinate, i.e. is directly
//		below or above newPoint.
//		See lecture slide on measuring heights.
//
// HINT1: make sure to dehomogenize points when necessary
// HINT2: there is a degeneracy that you should look out for involving points already in line with the reference
// HINT3: make sure to get the sign of the result right, i.e. whether it is above or below ground
// HINT4: check for some helpful/necessary variables which are listed in ImgView.h such as the vanishing points and homography H
void ImgView::sameXY()
{
	/*TSEARS Notes
	What does HINT2 Mean?
	This is incomplete, I need to project from the "reference plane" to the image plane.
	*/
	if (pntSelStack.size() < 2)
	{
		fl_alert("Not enough points on the stack.");
		return;
	}

	SVMPoint &newPoint = *pntSelStack[pntSelStack.size() - 1];
	SVMPoint &knownPoint = *pntSelStack[pntSelStack.size() - 2];

	if( !knownPoint.known() )
	{
		fl_alert("Can't compute relative values for unknown point.");
		return;
	}

	if( refPointOffPlane == NULL )
	{
		fl_alert("Need to specify the reference height first.");
		return;
	}

	/******** BEGIN TODO ********/

	// See the lecture note on measuring heights
	// using a known point directly below the new point.

	// printf("sameXY() to be implemented!\n");


    //TODO-BLOCK-BEGIN
    //Compute the horizon by crossing x and y
    SVMPoint horizon = xVanish.image_cross(yVanish); 
    horizon.image_dehomog();

    //Make a placeholder line and point
    SVMPoint line,line2;
    SVMPoint point,bpoint;


    //Find the reference point on the plane
    bpoint = *refPointOffPlane;
    printf("refPointOffPlane x %f y %f z %f u %f v %f\n",
    				bpoint.X,
    				bpoint.Y,
    				bpoint.Z,
    				bpoint.u,
    				bpoint.v);
    ApplyHomography(bpoint.u, bpoint.v, H, bpoint.X/bpoint.Z, bpoint.Y/bpoint.Z, 1);
    printf("refPointOffPlane after Homog x %f y %f z %f u %f v %f\n",
    				bpoint.X,
    				bpoint.Y,
    				bpoint.Z,
    				bpoint.u,
    				bpoint.v);
    //Compute the line from the reference point to the known point
    //to the horizon
    line = knownPoint.image_cross(bpoint);
    line.image_dehomog();

    //Compute the intersection of this line with the horizon
    point = line.image_cross(horizon);
    point.image_dehomog();

    //Now we compute the line from this horizon point to the new point
    line = newPoint.image_cross(horizon);
    line.image_dehomog();

    //We need to find a line from the refrence point to the z vanishing point
    line2 = zVanish.image_cross(*refPointOffPlane);

    //Now we find the point at which these two lines intersect
    point = line.image_cross(line2);
    point.image_dehomog();

    //Finally we can find the disance to the reference point
    double distance;
    distance = point.image_diff(bpoint).image_mag();
    distance/=bpoint.image_diff(*refPointOffPlane).image_mag();
    distance*=referenceHeight;

    newPoint.X=knownPoint.X;
    newPoint.Y=knownPoint.Y;
    newPoint.Z=distance;


    
    //printf("TODO: %s:%d\n", __FILE__, __LINE__);
    //TODO-BLOCK-END
	/******** END TODO ********/

	newPoint.known(true);

	printf( "Calculated new coordinates for point: (%e, %e, %e)\n", newPoint.X, newPoint.Y, newPoint.Z );

	redraw();
}



//
// TODO 5: sameZPlane()
//		Compute the 3D position of newPoint using knownPoint
//		that lies on the same plane and whose 3D position is known.
//		See the man on the box lecture slide.
//		If newPoint is on the reference plane (Z==0), use homography (this->H, or simply H) directly.
//
// HINT: For this function, you will only need to use the three vanishing points and the reference homography 
//       (in addition to the known 3D location of knownPoint, and the 2D location of newPoint)
void ImgView::sameZPlane()
{
	if (pntSelStack.size() < 2)
	{
		fl_alert("Not enough points on the stack.");
		return;
	}

	SVMPoint &newPoint = *pntSelStack[pntSelStack.size() - 1];
	SVMPoint &knownPoint = *pntSelStack[pntSelStack.size() - 2];

	if( !knownPoint.known() )
	{
		fl_alert("Can't compute relative values for unknown point.");
		return;
	}

	/******** BEGIN TODO ********/
    //TODO-BLOCK-BEGIN
    printf("TODO: %s:%d\n", __FILE__, __LINE__);
    //TODO-BLOCK-END
	/******** END TODO ********/

	newPoint.known(true);

	printf( "Calculated new coordinates for point: (%e, %e, %e)\n", newPoint.X, newPoint.Y, newPoint.Z );

	redraw();
}


