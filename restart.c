#include <stdio.h>
#include <string.h>

/***** Function to write wavefunction in binary format ************/
/*      Write Psi in binary format                                     */
/*      Input :         const int k, loop step index k used in filename.*/
/* Output:      CMPLX* Psi, wavefunction array                          */
/*                      REAL E0, initial energy for this wavefunction   */
/*              In the caller, intended to be used as:                 */
/*              if(k==300) save_wavefunction(k, Psi, E0);                   */
/*              (see example above)                                    */
/******************************************************************/
void save_wavefunction(const int NG, const int k, CMPLX* Psi, REAL E0)
{       printf("Now Saving ...");
        char str_Psi[20];
        #ifndef _UFM_
        sprintf(str_Psi,"SAVE/%08d.bin",k);
        char str_EN0[] = "SAVE/EN0.bin";
        #else
        sprintf(str_Psi,"save/%08d.bin",k);
        char str_EN0[] = "save/EN0.bin";
        #endif
        //printf("check str_EN0 = %s\n", str_EN0);
        FILE *fPsi, *fEN0;
        if (( fPsi = fopen(str_Psi, "wb") ) == NULL )
        {       printf(" file open error!!__%s", str_Psi);      }
        else
        {       fwrite(Psi, sizeof(CMPLX), NG, fPsi); fclose(fPsi);}
        if (( fEN0 = fopen(str_EN0, "wb") ) == NULL )
        {       printf(" file open error!!__%s", str_EN0);      }
        else
        {       fwrite(&E0, sizeof(REAL),   1, fEN0); fclose(fEN0); }
        //      Exit loop k and finalize once the binary file is written
        printf(" done!\n");
}

/**** Function to load wavefunction *************************/
/*      Load the "*.bin" file with the largest step index stored in  */
/*      directory SAVE (or save) into the wavefunction array           */
/*              Input : int N_t_step, total number of time-step loops  */
/*                                      int NG, total grid points of wavefunction */
/*                                      CMPLX*, Psi wavefunction to load      */
/*                                      REAL* E0, initial energy to load      */
/*              Output: int (return), load status, 0: false, 1: true   */
/*                                      int* k_start, starting step k on restart*/
/*      If a wavefunction file is found, values are loaded into Psi and E0 */
/*      If not found, Psi and E0 remain unchanged                      */
/*************************************************************/
int load_wavefunction(const int N_t_step, const int NG, int* k_start, CMPLX* Psi, REAL* E0)
{ /* Search for "000~~~~~.bin" in SAVE (or save) and restart from it if found */
        FILE *fPsi, *fEN0;
        #ifndef _UFM_
        char str_EN0[] = "SAVE/EN0.bin";
        #else
        char str_EN0[] = "save/EN0.bin";
        #endif
        char str_Psi[20];
        printf("Searching for %s\n", str_EN0);
        if ( (fEN0 = fopen(str_EN0, "rb")) == NULL )
        {       printf("Initial Energy is NOT found.\n");
                return 0; // False
        }
        for (int k = N_t_step; k >= 0; k--)     // check in descending order of k
        {       
                #ifndef _UFM_
                sprintf(str_Psi, "SAVE/%08d.bin", k);
                #else
                sprintf(str_Psi, "save/%08d.bin", k);
                #endif
                printf("Searching for %s\r", str_Psi);
                if ( (fPsi = fopen(str_Psi, "rb")) != NULL )
                {       //      if fopen does not return NULL,
                        //      i.e., if the file is found
                        printf("Savedata is found: %s\n", str_Psi);
                        fread(Psi, sizeof(CMPLX), NG, fPsi);
                        fread(E0,  sizeof(REAL),   1, fEN0);
                        fclose(fPsi);fclose(fEN0);
                        printf(" done!\n");
                        *k_start = k + 1;       // start from the step immediately following the saved data
                        return 1; // Ture
                }
        }
        printf("00~~~~~~.bin file is NOT found.\n");
        return 0; // False
}
