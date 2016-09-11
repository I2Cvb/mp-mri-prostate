C++ pre-processing
==================

What is in there?
-----------------

* `reg_dce`: registration of DCE volume.

### `reg_dce`

* `reg_dce.cxx`: ITK script to register each serie in the DCE modality. The metric used is the Mattes mutual information optimized through a regular step gradient descent. Only a rigid registration is performed.
* `reg_gt.cxx`: ITK script to register the ground-truth of T2W and DCE and apply the transform found on the DCE series. The metric used is the the mean squared metric optimized through a regular step gradient descent. We used a rigid registration follow by a coarse and fine bspline deformable registration.

