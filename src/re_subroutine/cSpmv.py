"""
cSparseMatVec
==============================================
KERNEL void cCSR_spmv(    
      const    uint    numRow,
      GLOBAL_MEM const uint *rowDelimiters, 
      GLOBAL_MEM const uint *cols,
      GLOBAL_MEM const float2 *val,
      GLOBAL_MEM const float2 *vec, 
      GLOBAL_MEM float2 *out)
      
Offload Sparse Matrix Vector Multiplication to heterogeneous devices.
Note: In CUDA, += operator can cause problems. Here we use explicit add operator.
"""

R="""


KERNEL void cCSR_spmv_vector(    
      const    uint    numRow,
      GLOBAL_MEM const uint *rowDelimiters, 
      GLOBAL_MEM const uint *cols,
      GLOBAL_MEM const float2 *val,
      GLOBAL_MEM const float2 *vec, 
      GLOBAL_MEM float2 *out)
{   
    const uint t = get_local_id(0);
    const uint vecWidth=${LL};
    // Thread ID within wavefront
    const uint id = t & (vecWidth-1);
    // One row per wavefront
    uint vecsPerBlock=get_local_size(0)/vecWidth;
    uint myRow=(get_group_id(0)*vecsPerBlock) + (t/ vecWidth);
    LOCAL_MEM float2 partialSums[${LL}];
    float2 zero;
    zero.x = 0.0;
    zero.y = 0.0;
    
    partialSums[t] = zero;
    float2  y= zero;
    if (myRow < numRow)
    {
     const uint vecStart = rowDelimiters[myRow];
     const uint vecEnd = rowDelimiters[myRow+1];            
     for (uint j = vecStart+id;  j<vecEnd; j += vecWidth)
     {
          const uint col = cols[j];
          const float2 spdata=val[j];
          const float2 vecdata=vec[col];                        
          y.x=spdata.x*vecdata.x - spdata.y*vecdata.y;
          y.y=spdata.y*vecdata.x + spdata.x*vecdata.y;
          partialSums[t] = partialSums[t] + y;
      }

      LOCAL_BARRIER; 
      //__syncthreads();
      //barrier(CLK_LOCAL_MEM_FENCE);
      // Reduce partial sums
      uint bar = vecWidth / 2;
      while(bar > 0)
      {
           if (id < bar)
          partialSums[t] = partialSums[t] + partialSums[t+bar];
           
           //barrier(CLK_LOCAL_MEM_FENCE);
           //__syncthreads();
           LOCAL_BARRIER;
           bar = bar / 2;
      }            
      // Write result 
      if (id == 0)
      {
       out[myRow]=partialSums[t]; 
      }
     }
};    // End of cCSR_spmv_vector
 
KERNEL void pELL_spmv_scalar(    
      const    uint    nRow,        // number of rows
      const    uint    prodJd,     // product of Jd
      const    uint    sumJd,     // sum of Jd
      const    uint    dim,           // dimensionality
      GLOBAL_MEM const uint *Jd,    // Jd
      // GLOBAL_MEM const uint *curr_sumJd,    // 
      GLOBAL_MEM const uint *meshindex,    // meshindex, prodJd * dim
      GLOBAL_MEM const uint *kindx,    // unmixed column indexes of all dimensions
      GLOBAL_MEM const float2 *udata,    // interpolation data before Kronecker product
      GLOBAL_MEM const float2 *vec,     // kspace data
      GLOBAL_MEM float2 *out)   // output
{      
    uint myRow= get_global_id(0);
    float2 zero;
    zero.x = 0.0;
    zero.y = 0.0;
    if (myRow < nRow)
    {
         float2 y = zero;
         for (uint j = 0;  j  <  prodJd; j ++)
         {    
            // now doing the first dimension
            uint index_shift = myRow * sumJd;
            // uint tmp_sumJd = 0;
            uint J = Jd[0];
            uint index =    index_shift +  meshindex[dim*j + 0];
            uint col = kindx[index] ;
            float2 spdata = udata[index];
            index_shift += J; 
            for (uint dimid = 1; dimid < dim; dimid ++ )
            {
                J = Jd[dimid];
                index =   index_shift + meshindex[dim*j + dimid];   // the index of the partial ELL arrays *kindx and *udata
                col += kindx[index] + 1  ;                                            // the column index of the current j
                float tmp_x = spdata.x;
                float2 tmp_udata = udata[index];
                spdata.x = spdata.x * tmp_udata.x - spdata.y * tmp_udata.y;                            // the spdata of the current j
                spdata.y = tmp_x * tmp_udata.y + spdata.y * tmp_udata.x; 
                index_shift  += J;
            }; // Iterate over dimensions 1 -> Nd - 1
            float2 vecdata=vec[col ];
            y.x +=  spdata.x*vecdata.x - spdata.y*vecdata.y;
            y.y +=  spdata.y*vecdata.x + spdata.x*vecdata.y;
        }; // Iterate for (uint j = 0;  j  <  prodJd; j ++)
        out[myRow]= y;
    };  // if (myRow < nRow)
};    // End of pELL_spmv_scalar
    
KERNEL void pELL_spmv_vector(    
      const    uint    nRow,        // number of rows
      const    uint    prodJd,     // product of Jd
      const    uint    sumJd,     // sum of Jd
      const    uint    dim,           // dimensionality
      GLOBAL_MEM const uint *Jd,            // Jd
      //GLOBAL_MEM const uint *curr_sumJd,            // 
      GLOBAL_MEM const uint *meshindex,            // meshindex, prodJd * dim
      GLOBAL_MEM const uint *kindx,    // unmixed column indexes of all dimensions
      GLOBAL_MEM const float2 *udata,// interpolation data before Kronecker product
      GLOBAL_MEM const float2 *vec,     // kspace data
      GLOBAL_MEM float2 *out)   // output
{   
    const uint t = get_local_id(0);
    const uint vecWidth=${LL};
    // Thread ID within wavefront
    const uint id = t & (vecWidth-1);
    
    // One row per wavefront
    uint vecsPerBlock=get_local_size(0)/vecWidth;
    uint myRow=(get_group_id(0)*vecsPerBlock) + (t/ vecWidth);
    LOCAL_MEM float2 partialSums[${LL}];
    float2 zero;
    zero.x = 0.0;
    zero.y = 0.0;
    partialSums[t] = zero;
    
    if (myRow < nRow)
    {
     const uint vecStart = 0; 
     const uint vecEnd =prodJd;             
     float2  y;//=zero;
     
     for (uint j = vecStart+id;  j<vecEnd; j += vecWidth)
     {    // now doing the first dimension
        uint J = Jd[0];
        uint index_shift = myRow * sumJd ;
        uint index =    index_shift +  meshindex[dim*j + 0];
        uint col = kindx[index] ;
        float2 spdata = udata[index];
        
        index_shift += J; 
        
        for (uint dimid = 1; dimid < dim; dimid ++ )
        {
            uint J = Jd[dimid];
            uint index =  index_shift + meshindex[dim*j + dimid];   // the index of the partial ELL arrays *kindx and *udata
            col += kindx[index] + 1;                                            // the column index of the current j
            float tmp_x= spdata.x;
            float2 tmp_udata = udata[index];
            spdata.x = spdata.x * tmp_udata.x - spdata.y * tmp_udata.y;                            // the spdata of the current j
            spdata.y = tmp_x * tmp_udata.y + spdata.y * tmp_udata.x; 
            index_shift  += J;
        }
        float2 vecdata=vec[col ];
        y.x =  spdata.x*vecdata.x - spdata.y*vecdata.y;
        y.y =  spdata.y*vecdata.x + spdata.x*vecdata.y;
        partialSums[t] = y + partialSums[t];
        
      }

      LOCAL_BARRIER; 

      // Reduce partial sums
      uint bar = vecWidth / 2;
      while(bar > 0)
      {
           if (id < bar)
          partialSums[t]= partialSums[t]+partialSums[t+bar];
           
           //barrier(CLK_LOCAL_MEM_FENCE);
           //__syncthreads();
           LOCAL_BARRIER;
           bar = bar / 2;
      }            
      // Write result 
      if (id == 0)
      {
       out[myRow]=partialSums[t]; 
      }
     }
};  // End of pELL_spmv_vector
    

KERNEL void pELL_spmv_mCoil(    
     const     uint   Reps,            // Number of coils
      const    uint    nRow,        // number of rows
      const    uint    prodJd,     // product of Jd
      const    uint    sumJd,     // sum of Jd
      const    uint    dim,           // dimensionality
      GLOBAL_MEM const uint *Jd,            // Jd, length = dim
      //GLOBAL_MEM const uint *curr_sumJd,            // summation of Jd[0:dimid] 
      GLOBAL_MEM const uint *meshindex,            // meshindex, prodJd * dim
      GLOBAL_MEM const uint *kindx,    // unmixed column indexes of all dimensions
      GLOBAL_MEM const float2 *udata,// interpolation data before Kronecker product
      GLOBAL_MEM const float2 *vec,     // multi-channel kspace data, prodKd * Reps
      GLOBAL_MEM float2 *out)   // multi-channel output, nRow * Reps
{   
    const uint t = get_local_id(0);
    const uint vecWidth=${LL};
    // Thread ID within wavefront
    const uint id = t & (vecWidth-1);
    
    // One row per wavefront
    uint vecsPerBlock=get_local_size(0)/vecWidth;
    uint myRow=(get_group_id(0)*vecsPerBlock) + (t/ vecWidth); // the myRow-th non-Cartesian sample
    uint m = myRow / Reps;
    uint nc = myRow - m * Reps;
    LOCAL_MEM float2 partialSums[${LL}];
    float2 zero;
    zero.x = 0.0;
    zero.y = 0.0;
    partialSums[t] = zero;
    
    if (myRow < nRow * Reps)
    {
     const uint vecStart = 0; 
     const uint vecEnd =prodJd;             
     float2  y;//=zero;
     
     for (uint j = vecStart+id;  j<vecEnd; j += vecWidth)
     {    // now doing the first dimension
        uint J = Jd[0];
        uint index_shift = m * sumJd ;
        uint index =    index_shift +  meshindex[dim*j + 0];
        uint col = kindx[index] ;
        float2 spdata = udata[index];
        
        index_shift += J; 
        
        for (uint dimid = 1; dimid < dim; dimid ++ )
        {
            uint J = Jd[dimid];
            uint index =  index_shift + meshindex[dim*j + dimid];   // the index of the partial ELL arrays *kindx and *udata
            col += kindx[index] + 1;                                            // the column index of the current j
            float tmp_x= spdata.x;
            float2 tmp_udata = udata[index];
            spdata.x = spdata.x * tmp_udata.x - spdata.y * tmp_udata.y;                            // the spdata of the current j
            spdata.y = tmp_x * tmp_udata.y + spdata.y * tmp_udata.x; 
            index_shift  += J;
        }
        float2 vecdata=vec[col * Reps + nc];
        y.x =  spdata.x*vecdata.x - spdata.y*vecdata.y;
        y.y =  spdata.y*vecdata.x + spdata.x*vecdata.y;
        partialSums[t] = y + partialSums[t];
        
      }

      LOCAL_BARRIER; 

      // Reduce partial sums
      uint bar = vecWidth / 2;
      while(bar > 0)
      {
           if (id < bar)
          partialSums[t]= partialSums[t]+partialSums[t+bar];
           
           //barrier(CLK_LOCAL_MEM_FENCE);
           //__syncthreads();
           LOCAL_BARRIER;
           bar = bar / 2;
      }            
      // Write result 
      if (id == 0)
      {
       out[myRow]=partialSums[t]; 
      }
     }
};  // End of pELL_spmv_mCoil

KERNEL void cELL_spmv_vector(    
            const    uint    numRow,
            const    uint    prodJd,
            GLOBAL_MEM const uint *cols,
            GLOBAL_MEM const float2 *val,
            GLOBAL_MEM const float2 *vec, 
            GLOBAL_MEM float2 *out)
{   
    const uint t = get_local_id(0);
    const uint vecWidth=${LL};
    // Thread ID within wavefront
    const uint id = t & (vecWidth-1);
    // One row per wavefront
    uint vecsPerBlock=get_local_size(0)/vecWidth;
    uint myRow=(get_group_id(0)*vecsPerBlock) + (t/ vecWidth);
    LOCAL_MEM float2 partialSums[${LL}];
    float2 zero;
    zero.x = 0.0;
    zero.y = 0.0;
    
    partialSums[t] = zero;
    float2  y= zero;
    if (myRow < numRow)
    {
        const uint vecStart = myRow*prodJd; 
        const uint vecEnd = (myRow + 1)*prodJd;             
        for (uint j = vecStart+id;  j<vecEnd; j += vecWidth)
        {
            const uint col    =    cols[j];
            const float2 spdata    =    val[j];
            const float2 vecdata    =    vec[col];                        
            y.x=spdata.x*vecdata.x - spdata.y*vecdata.y;
            y.y=spdata.y*vecdata.x + spdata.x*vecdata.y;
            partialSums[t] = y + partialSums[t];
        }
        LOCAL_BARRIER; 
        // Reduce partial sums
        uint bar = vecWidth / 2;
        while(bar > 0)
        {
             if (id < bar)
            {
                partialSums[t]= partialSums[t]+partialSums[t+bar];
            }
            LOCAL_BARRIER;
            bar = bar / 2;
        }            
        // Write result 
        if (id == 0)
        {
        out[myRow]=partialSums[t]; 
        }
    }
};    // End of cELL_spmv_vector

KERNEL void cELL_spmv_scalar(    
      const    uint    nRow,
      const    uint    prodJd, 
      GLOBAL_MEM const uint *cols,
      GLOBAL_MEM const float2 *data,
      GLOBAL_MEM const float2 *vec, 
      GLOBAL_MEM float2 *out)
{   
    uint myRow= get_global_id(0);
    float2 zero;
    zero.x = 0.0;
    zero.y = 0.0;
    if (myRow < nRow)
    {      
         float2  y= zero;
         out[myRow] =zero; 
         for (uint j = myRow *  prodJd;  j< (myRow + 1) *  prodJd; j ++)
         {
              uint col = cols[j];
              float2 spdata=data[j];
              float2 vecdata=vec[col];                        
              y.x +=spdata.x*vecdata.x - spdata.y*vecdata.y;
              y.y +=spdata.y*vecdata.x + spdata.x*vecdata.y;
         }
         //LOCAL_BARRIER not required here
        out[myRow]= y;
    } // if (myRow < nRow)
}; // End of cELL_spmv_scalar       
      
    
"""
from numpy import uint32
scalar_arg_dtypes=[uint32, None, None, None, None, None]        
