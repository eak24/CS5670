/* iScissor.cpp */
/* Main file for implementing project 1.  See TODO statments below
 * (see also correlation.cpp and iScissor.h for additional TODOs) */

#include <assert.h>
#include <stdlib.h>     /* calloc, exit, free */
#include <stdio.h>      /* printf, scanf, NULL */
#include "correlation.h"
#include "iScissor.h"
#include <queue>
using namespace std;  // This is to make available the names of things defined in the standard library.


const double linkLengths[8] = { 1.0, SQRT2, 1.0, SQRT2, 1.0, SQRT2, 1.0, SQRT2 };

// two inlined routines that may help;

inline Node& NODE(Node* n, int i, int j, int width)
{
    return *(n + j * width + i);
}

inline unsigned char PIXEL(const unsigned char* p, int i, int j, int c, int width)
{
    return *(p + 3 * (j * width + i) + c);
}

/************************ TODO 1 ***************************
 *InitNodeBuf
 *	INPUT:
 *		img:	a RGB image of size imgWidth by imgHeight;
 *		nodes:	a allocated buffer of Nodes of the same size, one node corresponds to a pixel in img;
 *  OUPUT:
 *      initializes the column, row, and linkCost fields of each node in the node buffer.
 */

/*TIM AND EAK's NOTES
This function initializes the node buffer. It should use each of the eight kernel in the 
kernels vector to fill the links[] array in each of the Nodes[]. 
*/
void InitNodeBuf(Node* nodes, const unsigned char* img, int imgWidth, int imgHeight)
{
    //double * cost = (double *) calloc(imgHeight*imgWidth*3,sizeof(double));
    double pixel[3]; //
    double maxD=0.0;
    double LENGTHS[2] = {1,SQRT2}; 
    int k,c,x,y; 
    for (x=0;x<imgWidth;x++)
        for(y=0;y<imgHeight;y++)
            for(k=0; k<8; k++){
                pixel_filter(pixel, x, y,img, imgWidth, imgHeight,
                    kernels[k], 3,3,1,0);
                nodes[y*imgWidth+x].linkCost[k]=sqrt(pixel[0]*pixel[0]+pixel[1]*pixel[1]+pixel[2]*pixel[2]);
                nodes[y*imgWidth+x].state=INITIAL;
                nodes[y*imgWidth+x].row=y;
                nodes[y*imgWidth+x].column=x;
                maxD = max(maxD,nodes[y*imgWidth+x].linkCost[k]);
    }
        
    for (x=0;x<imgWidth;x++)
                for(y=0;y<imgHeight;y++)
                    for(k=0;k<8;k++)
                        nodes[y*imgWidth+x].linkCost[k] = (maxD - nodes[y*imgWidth+x].linkCost[k])/LENGTHS[k & 1];
                    
}
/************************ END OF TODO 1 ***************************/

static int offsetToLinkIndex(int dx, int dy)
{
    int indices[9] = { 3, 2, 1, 4, -1, 0, 5, 6, 7 };
    int tmp_idx = (dy + 1) * 3 + (dx + 1);
    assert(tmp_idx >= 0 && tmp_idx < 9 && tmp_idx != 4);
    return indices[tmp_idx];
}

//Compare function for the priority queue. Requires ordered 2 nodes. Returns 
//true if n1 total cost is greater than n2's total cost.
class CompareNode {
    public:
    bool operator()(Node * n1, Node * n2) 
    {
       if (n1->totalCost > n2->totalCost) return true;
       return false;
    }
};

/************************ TODO 4 ***************************
 *LiveWireDP:
 *	INPUT:
 *		seedX, seedY:	seed position in nodes
 *		nodes:			node buffer of size width by height;
 *      width, height:  dimensions of the node buffer;
 *		selection:		if selection != NULL, search path only in the subset of nodes[j*width+i] if selection[j*width+i] = 1;
 *						otherwise, search in the whole set of nodes.
 *		numExpanded:		compute the only the first numExpanded number of nodes; (for debugging)
 *	OUTPUT:
 *		computes the minimum path tree from the seed node, by assigning
 *		the prevNode field of each node to its predecessor along the minimum
 *		cost path from the seed to that node.
 */

void LiveWireDP(int seedX, int seedY, Node* nodes, int width, int height, 
    const unsigned char* selection, int numExpanded)
{
    printf("(LiveWireDP)\n");
    Node *q,*r;
    int dir,offsetX,offsetY; 
    //double i=0;
    //initialize priority queue pq to be empty and set each node to INITIAL state
    for (dir=0;dir<(height*width);dir++){
        nodes[dir].state=INITIAL;
        nodes[dir].totalCost=0;
        nodes[dir].prevNode=NULL;
    }
    priority_queue<Node*, vector<Node*>, CompareNode> pq;
    
    //set seed total cost to 0 and insert seed into queue
    nodes[seedX+width*seedY].totalCost=0.0;
    pq.push(&nodes[seedX+width*seedY]);

    while (!pq.empty()){
        q = pq.top();
        //printf("this totalcost is %f\n",q->totalCost);
        pq.pop();
        //i+=1;
        //printf("(Work) %f\n",(100*i)/(width*height) );
        q->state = EXPANDED;

        //for each valid neighbor, r of node q
        for (dir=0;dir<8;dir++){
            q->nbrNodeOffset(offsetX,offsetY,dir);
            offsetX+=q->column; offsetY+=q->row;

            //set r as neighbor node if r is still in image
            if (!(offsetX<0||offsetX>=width||offsetY<0||offsetY>=height)){
                r=&(nodes[offsetY*width+offsetX]);

            //if r is INITIAL, insert r to pq and set r's total cost to the sum of
            // the total cost of q and the link cost from q to r... mark r as 
            //active and insert into pq
                if (r->state==INITIAL){
                    r->totalCost = q->totalCost+q->linkCost[dir];
                    r->state  =ACTIVE;
                    r->prevNode = q;
                    pq.push(r);
                }

                //if r is active then test for less expensive path and update if 
                //one exists using the link from q to r
                if(r->state==ACTIVE){
                    if (r->totalCost>(q->totalCost+q->linkCost[dir])){
                            r->prevNode = q; 
                            r->totalCost=(q->totalCost+q->linkCost[dir]);
                    }
                }
            }
        }
    }
        printf("(LiveWireDPdone)\n");

}
/************************ END OF TODO 4 ***************************/

/************************ TODO 5 ***************************
 *MinimumPath:
 *	INPUT:
 *		nodes:				a node buffer of size width by height;
 *		width, height:		dimensions of the node buffer;
 *		freePtX, freePtY:	an input node position;
 *	OUTPUT:
 *		insert a list of nodes along the minimum cost path from the seed node to the input node.
 *		Notice that the seed node in the buffer has a NULL predecessor.
 *		And you want to insert a *pointer* to the Node into path, e.g.,
 *		insert nodes+j*width+i (or &(nodes[j*width+i])) if you want to insert node at (i,j), instead of nodes[nodes+j*width+i]!!!
 *		after the procedure, the seed should be the head of path and the input code should be the tail
 */

void MinimumPath(CTypedPtrDblList <Node>* path, int freePtX, int freePtY, Node* nodes, int width, int height)
{
    Node* curr = &(nodes[freePtX+freePtY*width]);

        while (curr!=NULL) {
            path->AddHead (curr);
            curr = curr->prevNode;
        }
}
/************************ END OF TODO 5 ***************************/

/************************ An Extra Credit Item ***************************
 *SeeSnap:
 *	INPUT:
 *		img:				a RGB image buffer of size width by height;
 *		width, height:		dimensions of the image buffer;
 *		x,y:				an input seed position;
 *	OUTPUT:
 *		update the value of x,y to the closest edge based on local image information.
 */

void SeedSnap(int& x, int& y, unsigned char* img, int width, int height)
{
    printf("SeedSnap in iScissor.cpp: to be implemented for extra credit!\n");
}

//generate a cost graph from original image and node buffer with all the link costs;
void MakeCostGraph(unsigned char* costGraph, const Node* nodes, const unsigned char* img, int imgWidth, int imgHeight)
{
    int graphWidth = imgWidth * 3;
    int graphHeight = imgHeight * 3;
    int dgX = 3;
    int dgY = 3 * graphWidth;

    int i, j;
    for (j = 0; j < imgHeight; j++) {
        for (i = 0; i < imgWidth; i++) {
            int nodeIndex = j * imgWidth + i;
            int imgIndex = 3 * nodeIndex;
            int costIndex = 3 * ((3 * j + 1) * graphWidth + (3 * i + 1));

            const Node* node = nodes + nodeIndex;
            const unsigned char* pxl = img + imgIndex;
            unsigned char* cst = costGraph + costIndex;

            cst[0] = pxl[0];
            cst[1] = pxl[1];
            cst[2] = pxl[2];

            //r,g,b channels are grad info in seperate channels;
            DigitizeCost(cst	   + dgX, node->linkCost[0]);
            DigitizeCost(cst - dgY + dgX, node->linkCost[1]);
            DigitizeCost(cst - dgY      , node->linkCost[2]);
            DigitizeCost(cst - dgY - dgX, node->linkCost[3]);
            DigitizeCost(cst	   - dgX, node->linkCost[4]);
            DigitizeCost(cst + dgY - dgX, node->linkCost[5]);
            DigitizeCost(cst + dgY	   ,  node->linkCost[6]);
            DigitizeCost(cst + dgY + dgX, node->linkCost[7]);
        }
    }
}

