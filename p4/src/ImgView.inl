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
	What does HINT2 Mean? Maybe done
    Am I gaurunteed that knownpoint is on the reference plane?
    Please look at the line marked ATTENTION
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

    ///First comptue the homographies????
    //ComputeHomography(H, Hinv, points, vector<Vec3d> &basisPts, bool isRefPlane)
    //Compute the horizon by crossing x and y
    SVMPoint horizon = xVanish.image_cross(yVanish); 
    horizon.image_dehomog();

    //Make a placeholder line and point
    SVMPoint line,line2;
    SVMPoint point,bpoint;


    //Find the reference point on the plane
    bpoint = *refPointOffPlane;


    //bpoint.u = H[0][0] * bpoint.X+ H[0][1] * bpoint.Y;
    //bpoint.v = H[1][0] * bpoint.X + H[1][1] * bpoint.Y;
    ApplyHomography(bpoint.u, bpoint.v, H, bpoint.X, bpoint.Y, 1);

    line = refPointOffPlane->image_cross(zVanish);
    point = line.image_cross(newPoint);
    if(point.u==0 && point.v==0){
        point = newPoint;
        printf("degeneracy, newpoint is in line with the refrence point\n");
    }
    else{
    //Make sure the known point is on the reference plane.
    
    //Should I do this?? ATTENTION
    printf("known point uv %fx%f, xyz %fx%fx%f\n",knownPoint.u,knownPoint.v,knownPoint.X,knownPoint.Y,knownPoint.Z);
    ApplyHomography(point.u,point.v,H,knownPoint.X/knownPoint.W,knownPoint.Y/knownPoint.W,1);
    printf("point on plane after homog uv %fx%f\n",point.u,point.v);

    //Compute the line from the reference point to the known point
    //to the horizon
    line = point.image_cross(bpoint);
    line.image_dehomog();

    //Compute the intersection of this line with the horizon
    point = line.image_cross(horizon);
    point.image_dehomog();

    printf("Base of reference to horizon. Intersection at %fx%f\n",point.u,point.v);

    //Now we compute the line from this horizon point to the new point
    line = newPoint.image_cross(point);
    line.image_dehomog();

    //We need to find a line from the refrence point to the z vanishing point
    line2 = zVanish.image_cross(*refPointOffPlane);

    //Now we find the point at which these two lines intersect
    point = line.image_cross(line2);
    point.image_dehomog();
    }   
    printf("Intersection of line from horizon to reference line is %fx%f\n",point.u,point.v);

    //Finally we can find the disance to the reference point
    double distance,distance2;
    distance = (point.u-bpoint.u)*(point.u-bpoint.u) + (point.v-bpoint.v)*(point.v-bpoint.v);
    distance = sqrt(distance); //Distance from intersection point to the ground plane
    printf("Image distance between intersection point and ground point %f\n",distance);
    distance2= (refPointOffPlane->u-bpoint.u)*(refPointOffPlane->u-bpoint.u) + (refPointOffPlane->v-bpoint.v)*(refPointOffPlane->v-bpoint.v);
    distance2 = sqrt(distance2);
    printf("Image distance between top of reference point and ground point %f\n",distance2);
    distance/=distance2;
    printf("ratio %f\n",distance);
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
    //Find the horixon based on the x and y vanishing points
    SVMPoint horizon = xVanish.image_cross(yVanish); 
    horizon.image_dehomog();

    //Connect the known point and the new point
    SVMPoint line = knownPoint.image_cross(newPoint);
    line.image_dehomog();

    //Find the intersection of the line with the horizon
    SVMPoint horizon_intersect = horizon.image_cross(line);
    horizon_intersect.image_dehomog();

    //Find the point on the reference plane beneath the known point
    SVMPoint bkpoint = knownPoint;
    ApplyHomography(bkpoint.u, bkpoint.v, H, bkpoint.X/bkpoint.W, bkpoint.Y/bkpoint.W, 1);

    //Find the line going to the base
    SVMPoint line_to_base = horizon_intersect.image_cross(bkpoint);
    line_to_base.image_dehomog();
    //Find a vertical line that goes through newpoint
    SVMPoint vertical_line = zVanish.image_cross(newPoint);
    vertical_line.image_dehomog();

    //Intersect line_to_base and vertical_line
    SVMPoint bqpoint = line_to_base.image_cross(vertical_line);

    ApplyHomography(newPoint.X,newPoint.Y, Hinv, bqpoint.u/bqpoint.w, bqpoint.v/bqpoint.w, 1);

    newPoint.Z = knownPoint.Z;

    //TODO-BLOCK-END
	/******** END TODO ********/

	newPoint.known(true);

	printf( "Calculated new coordinates for point: (%e, %e, %e)\n", newPoint.X, newPoint.Y, newPoint.Z );

	redraw();
}


