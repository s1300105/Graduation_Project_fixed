pixaWriteMemMultipageTiff(l_uint8  **pdata,
                          size_t    *psize,
                          PIXA      *pixa)
{
const char  *modestr;
l_int32      i, n;
FILE        *fp;
PIX         *pix1;

    PROCNAME("pixaWriteMemMultipageTiff");

    if (pdata) *pdata = NULL;
    if (!pdata)
        return ERROR_INT("pdata not defined", procName, 1);
    if (!pixa)
        return ERROR_INT("pixa not defined", procName, 1);

#ifdef _WIN32
    if ((fp = fopenWriteWinTempfile()) == NULL)
        return ERROR_INT("tmpfile stream not opened", procName, 1);
#else
    if ((fp = tmpfile()) == NULL)
        return ERROR_INT("tmpfile stream not opened", procName, 1);
#endif  /* _WIN32 */

    n = pixaGetCount(pixa);
    for (i = 0; i < n; i++) {
        modestr = (i == 0) ? "w" : "a";
        pix1 = pixaGetPix(pixa, i, L_CLONE);
        if (pixGetDepth(pix1) == 1)
            pixWriteStreamTiffWA(fp, pix1, IFF_TIFF_G4, modestr);
        else
            pixWriteStreamTiffWA(fp, pix1, IFF_TIFF_ZIP, modestr);
        pixDestroy(&pix1);
    }

    rewind(fp);
    *pdata = l_binaryReadStream(fp, psize);
    fclose(fp);
    return 0;
}