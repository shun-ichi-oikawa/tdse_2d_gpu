/******  Function to measure elapsed time in ms (start)   ****/
/*    Input:  pointer-to-pointer of StopWatchInterface   */
/*    Output: none                                       */
/*            Initialization only. In main(), declare:   */
/*			 StopWatchInterface* timer;               */
/*            as a timer variable, and pass it with &    */
/*			 SetTimer(&timer);                        */
/*           with & as the argument:                     */
/***************************************************/
void SetTimer(StopWatchInterface** TIMER)
{	
	sdkCreateTimer(TIMER);
	sdkStartTimer(TIMER); 
}

/****** Function to measure elapsed time in ms (stop)  ********/
/*    Input:  pointer-to-pointer of StopWatchInterface   */
/*   Output: (return) elapsed time in ms as double since */
/*           SetTimer was called                         */
/****** Function to measure elapsed time in ms (stop)  ********/
/*			 double time = EndTimer(&timer);           */
/*           with & as the argument:                     */
/****************************************************/
double EndTimer(StopWatchInterface** TIMER)
{	double tmp = -1;
	sdkStopTimer(TIMER);
	tmp = sdkGetTimerValue(TIMER);
	sdkDeleteTimer(TIMER);
	return tmp;
}
