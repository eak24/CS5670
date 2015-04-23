/***************************************************************
 * CS4670/5670, Fall 2012 Project 4
 * File to be modified #3:
 * ImgViewBox.cpp
 *		routines for finding the corners of an axis-aligned
 *      box (in 2D and in 3D)
 **************************************************************/
#pragma warning(disable : 4996)

#include "ImgView.h"

//
// TODO 6: solveForOppositeCorners()
//     Given the 2D positions of two corners of a rectangular face parallel to the XZ plane, compute
//     the 2D positions of the other two corners
void ImgView::solveForOppositeCorners(double u0, double v0, double u2, double v2,
                                      double &u1, double &v1, double &u3, double &v3)
{
    /* Vanishing points must be known */
    assert(xVanish.known() && yVanish.known() && zVanish.known());    
    /******** BEGIN TODO ********/ 
    // Given the 2D positions of corners p0 and p2 of the face, compute the 2D positions of p1 and p3
    // Remember that this face is on a plane perpendicular to the plane x=0
    // Store the results in variables 'u1, v1' and 'u3, v3'

    //TODO-BLOCK-BEGIN
    Vec3d p0 = Vec3d(u0,v0,1);
    Vec3d p2 = Vec3d(u2,v2,1);
    Vec3d xv = Vec3d(xVanish.u,xVanish.v,xVanish.w);
    Vec3d zv = Vec3d(zVanish.u,zVanish.v,zVanish.w);
    //Find the line connecting p0 and the xvanishing point
    Vec3d bottom_base = cross(p0,xv);

    //Find the line conneciing p2 and the xvanishing point
    Vec3d top_base = cross(p2,xv);

    //Find the vertical edge connecting 2 and the zvanishing point
    Vec3d far_edge = cross(p2,zv);

    //Find the vertical edge connecting 0 and the zvanishing point
    Vec3d near_edge = cross(p0,zv);

    Vec3d p1 = cross(bottom_base,far_edge);
    Vec3d p3 = cross(top_base,near_edge);
    p1/=p1[2];
    p3/=p3[2];
    u1 = p1[0];
    v1= p1[1];
    u3= p3[0];
    v3 = p3[1];


    //TODO-BLOCK-END
    /********* END TODO ********/
}

//
// TODO 7: solveForOppositeFace()
//     Given the 2D positions of one rectangular face parallel to the XZ plane, 
//     compute the 2D positions of a parallel face being swept out from it.
//     The mouse position is given; one of the lines on the parallel face should pass
//     through the mouse position
void ImgView::solveForOppositeFace(SVMSweep *sweep, double imgX, double imgY,
                                   Vec3d &p4_out, Vec3d &p5_out, Vec3d &p6_out, Vec3d &p7_out)
{
    SVMPolygon *poly = sweep->poly;

    if (poly == NULL)
        return;

    // Get the four existing points
    SVMPoint *n0, *n1, *n2, *n3;
    poly->getFourPoints(&n0, &n1, &n2, &n3);

    Vec3d p0(n0->u, n0->v, n0->w);
    Vec3d p1(n1->u, n1->v, n1->w);
    Vec3d p2(n2->u, n2->v, n2->w);
    Vec3d p3(n3->u, n3->v, n3->w);

    Vec3d pMouse(imgX, imgY, 1.0);

    /******** BEGIN TODO ********/
    // Find the 2D image positions of box corners p4, p5, p6, p7, as described on the webpage.  
    // You will compute these positions using the known corners of the box (p0, p1, p2, p3, defined above)
    // and the vanishing points.
    // The line through points p4 and p5 will go through the mouse position, pMouse
    // Store the results in variables p4, p5, p6, and p7.
	Vec3d p4, p5, p6, p7;

    //TODO-BLOCK-BEGIN
    Vec3d xv = Vec3d(xVanish.u,xVanish.v,xVanish.w);
    Vec3d zv = Vec3d(zVanish.u,zVanish.v,zVanish.w);
    Vec3d yv = Vec3d(yVanish.u,yVanish.v,yVanish.w);
    Vec3d mouse = Vec3d(imgX, imgY,1);
    //Find p5
    Vec3d line1 = cross(p1,yv);
    Vec3d line2 = cross(mouse, xv);
    p5 = cross(line1,line2);
    //Find p4
    line1 = cross(p0,yv);
    line2 = cross(mouse, xv);
    p4 = cross(line1,line2);
    //find p7
    line1 = cross(p4,zv);
    line2 = cross(p3,yv);
    p7 = cross(line1,line2);
    //find p6
    line1 = cross(p2,yv);
    line2 = cross(p5,zv);
    p6 = cross(line1,line2);

    //TODO-BLOCK-END

    /******** END TODO ********/

    p4_out = p4;
    p5_out = p5;
    p6_out = p6;
    p7_out = p7;
}

//
// TODO 8: find3DPositionsBox()
//    Find the 3D positions of the 8 corners of the box.  The 3D position of points[0] is known.
void ImgView::find3DPositionsBox(SVMPoint *points[8]) 
{
    /******** BEGIN TODO ********/
    // Implement this function.  You will compute the 3D positions of the corners of the box
    // using their 2D positions and the known 3D position of points[0].  Store the results in 
    // points[1] through points[7].  You can and should use the sameXY and sameZ routines that 
	// you need to implement.  For that to work, you will need to push and pop points from
	// pntSelStack.  There are multiple ways to implement this function.

    //TODO-BLOCK-BEGIN
    printf("TODO: %s:%d\n", __FILE__, __LINE__);
    //TODO-BLOCK-END

	/********* END TODO ********/
}


