
import numpy as np
import nibabel
from scipy import ndimage




# TODO: see if we can incorporate Kendrick's localregression as a scipy
# "generic filter".
class fnc_class:
    def __init__(self, weights):
        # store the shape and weights:
        self.weights = weights
        self.shape = weights.shape
        # initialize the coordinates:
        self.coords = [0] * len(weights.shape)

    def filter(self, buffer):
        ix = mgrid[max(0,ceil(self.coords[0]-buffer.shape[0])):min(self.shape[0]-1,floor(x0(pp)+buffer.shape[0]))]
        iy = mgrid[max(0,ceil(self.coords[1]-buffer.shape[1])):min(self.shape[1]-1,floor(y0(pp)+buffer.shape[1]))]
        iz = mgrid[max(0,ceil(self.coords[2]-buffer.shape[2])):min(self.shape[2]-1,floor(z0(pp)+buffer.shape[2]))]
        # WORK HERE
        result = (buffer * self.weights[ix,iy,iz]).sum()
        # calculate the next coordinates:
        axes = range(len(self.shape))
        axes.reverse()
        for jj in axes:
            if self.coords[jj] < self.shape[jj] - 1:
                self.coords[jj] += 1
                break
            else:
                self.coords[jj] = 0
        return result

# fnc = fnc_class(weights = fm_mag.get_data())
# generic_filter(a, fnc.filter, footprint = [[1, 0], [0, 1]])


def get_smoothed_fieldmap_localregression(fm, fmmag, epi_qform, epi_shape):
    xform = np.dot(np.linalg.inv(fm.get_qform()), epi_qform)
    [xxBc,yyBc,zzBc] = np.mgrid[0:epi_shape[0],0:epi_shape[1],0:epi_shape[2]] # grid the epi, get location of each sampled voxel
    # Now apply the transform. The following is equivalent to dot(xform,coords), where "coords" is the
    # list of all the (homogeneous) coords. Writing out the dot product like this is just faster and easier.
    xxB = xxBc*xform[0,0] + yyBc*xform[0,1] + zzBc*xform[0,2] + xform[0,3]
    yyB = xxBc*xform[1,0] + yyBc*xform[1,1] + zzBc*xform[1,2] + xform[1,3]
    zzB = xxBc*xform[2,0] + yyBc*xform[2,1] + zzBc*xform[2,2] + xform[2,3]
    # use local linear regression to smooth and interpolate the fieldmaps.
    fm_smooth_param = 7.5/np.array(fm.get_header().get_zooms()[0:3])  # Want 7.5mm in voxels, so =7.5/mm_per_vox
    fm_smooth = pytave.feval(1,'localregression3d',fm.get_data(),xxB+1,yyB+1,zzB+1,np.array([]),np.array([]),fm_smooth_param,fm_mag.get_data())[0]
    return fm_smooth

def get_smoothed_fieldmap(fm, fmmag, epi_qform, epi_shape):
    import pytave
    # FIXME: how do I automatically get the correct directory here?
    pytave.addpath('/home/bobd/git/nims/nimsutil')

    xform = np.dot(np.linalg.inv(fm.get_qform()), epi_qform)
    fm_pixdim = np.array(fm.get_header().get_zooms()[0:3])
    fm_mag_data = fm_mag.get_data().mean(3).squeeze()
    # clean up with a little median filter and some greyscale open/close.
    # We'll use a structure element that is about 5mm (rounded up to the nearest pixdim)
    filter_size = 5.0/fm_pixdim
    fm_mag_data = ndimage.median_filter(fm_mag_data, filter_size.round().astype(int))
    fm_mag_data = ndimage.morphology.grey_opening(fm_mag_data, filter_size.round().astype(int))
    fm_mag_data = ndimage.morphology.grey_closing(fm_mag_data, filter_size.round().astype(int))

    # Total image volume, in cc:
    fm_volume = np.prod(fm_pixdim) * np.prod(fm.get_shape()) / 1000
    # typical human cranial volume is up to 1800cc. There's also some scalp and maybe neck,
    # so we'll say 2500cc of expected tissue volume.
    mag_thresh = np.percentile(fm_mag_data, max(0.0,100.0*(1.0-2500.0/fm_volume)))
    mask = ndimage.binary_opening(fm_mag_data>mag_thresh, iterations=2)

    # Now delete all the small objects, just keeping the largest (which should be the brain!)
    label_im,num_objects = ndimage.measurements.label(mask)
    h,b = np.histogram(label_im,num_objects)
    mask = label_im==b[h==max(h[1:-1])]
    mask_volume = np.prod(fm_pixdim) * max(h[1:-1]) / 1000.0
    mask_sm = ndimage.gaussian_filter(mask.astype(np.float), filter_size)

    fm_Hz = fm.get_data().astype(np.float).squeeze()

    fm_Hz_sm = ndimage.gaussian_filter(fm_Hz * mask_sm, filter_size/2)

    fm_final = np.empty(epi_shape[0:3])
    ndimage.affine_transform(fm_Hz_sm, xform[0:3,0:3], offset=xform[0:3,3], output_shape=epi_shape[0:3], output=fm_final)

    [xxBc,yyBc,zzBc] = np.mgrid[0:epi_shape[0],0:epi_shape[1],0:epi_shape[2]] # grid the epi, get location of each sampled voxel
    # Now apply the transform. The following is equivalent to dot(xform,coords), where "coords" is the
    # list of all the (homogeneous) coords. Writing out the dot product like this is just faster and easier.
    xxB = xxBc*xform[0,0] + yyBc*xform[0,1] + zzBc*xform[0,2] + xform[0,3]
    yyB = xxBc*xform[1,0] + yyBc*xform[1,1] + zzBc*xform[1,2] + xform[1,3]
    zzB = xxBc*xform[2,0] + yyBc*xform[2,1] + zzBc*xform[2,2] + xform[2,3]
    # use local linear regression to smooth and interpolate the fieldmaps.
    fm_smooth_param = 7.5/np.array(fm.get_header().get_zooms()[0:3])  # Want 7.5mm in voxels, so =7.5/mm_per_vox
    fm_smooth = pytave.feval(1,'localregression3d',fm.get_data(),xxB+1,yyB+1,zzB+1,np.array([]),np.array([]),fm_smooth_param,fm_mag.get_data())[0]
    return fm_smooth


def fieldmap_unwarp(fmfile, fmmagfile, epifile):
    freq_dim,phase_dim,slice_dim = epi.get_header().get_dim_info()
    phase_encode_time = 0.0005
    phase_encode_acceleration = 2
    acquisition_matrix = [110,110,0]
    phase_encode_readout_direction = 1 # 1 or -1
    epi_readout_time = phase_encode_time*acquisition_matrix[phase_dim] / phase_encode_acceleration
    fm = nibabel.load(fmfile)
    fm_mag = nibabel.load(fmmagfile)
    epi = nibabel.load(epifile)
    fm_smooth = get_smoothed_fieldmap(fm, fmmag, epi.get_qform(), epi.shape)
    # The shift, in pixels, caused by the B0 error is:
    # B0 error (in Hz) * epi_readout_time (in seconds) * (epidim/epiinplanematrixsize)
    # TODO: decide if fm_scale should be negative! This depends on the readout direction.
    fm_scale = phase_encode_readout_direction * epi_readout_time * (float(epi.shape[phase_dim]) / acquisition_matrix[phase_dim])
    [xx,yy,zz] = np.mgrid[0:epi.shape[0],0:epi.shape[1],0:epi.shape[2]]
    warped_coords = np.array([xx,yy,zz], dtype='float')
    warped_coords[phase_dim,:,:,:] = warped_coords[phase_dim,:,:,:] + fm_smooth * fm_scale
    d = epi.get_data()
    unwarp = np.zeros(epi.shape[0:3], dtype=epi.get_data_dtype())
    for t in range(epi.shape[3]):
        ndimage.map_coordinates(d[:,:,:,t], warped_coords, unwarp, order=3)
        d[:,:,:,t] = unwarp

if __name__ == '__main__':
    # Image pyramid contains a nice image montage function:
    from nimsutil import pyramid

    epifile='/nimsfs/nims/ngolden/20120816_3167/0005_01_DTI_2mm_b2500_96dir.nii.gz'
    epi=nibabel.load(epifile)
    epi_shape=epi.shape
    epi_qform=epi.get_qform()

    fmfile = '/tmp/fm_B0.nii.gz'
    fmmagfile = '/tmp/fm.nii.gz'
    fm = nibabel.load(fmfile)
    fm_mag = nibabel.load(fmmagfile)
    fm_pixdim = np.array(fm.get_header().get_zooms()[0:3])
    fm_mag_data = fm_mag.get_data().mean(3).squeeze()
    # clean up with a little median filter and some greyscale open/close.
    # We'll use a structure element that is about 5mm (rounded up to the nearest pixdim)
    filter_size = 5.0/fm_pixdim
    fm_mag_data = ndimage.median_filter(fm_mag_data, filter_size.round().astype(int))
    fm_mag_data = ndimage.morphology.grey_opening(fm_mag_data, filter_size.round().astype(int))
    fm_mag_data = ndimage.morphology.grey_closing(fm_mag_data, filter_size.round().astype(int))
    pylab.imshow(pyramid.ImagePyramid(fm_mag_data).get_montage(), figure=pylab.figure(figsize=(16,16)))

    # Total image volume, in cc:
    fm_volume = np.prod(fm_pixdim) * np.prod(fm.get_shape()) / 1000
    # typical human cranial volume is up to 1800cc. There's also some scalp and maybe neck,
    # so we'll say 2500cc of expected tissue volume.
    mag_thresh = np.percentile(fm_mag_data, max(0.0,100.0*(1.0-2500.0/fm_volume)))
    mask = ndimage.binary_opening(fm_mag_data>mag_thresh, iterations=2)
    pylab.imshow(pyramid.ImagePyramid(mask).get_montage(), figure=pylab.figure(figsize=(16,16)))

    # Now delete all the small objects, just keeping the largest (which should be the brain!)
    label_im,num_objects = ndimage.measurements.label(mask)
    h,b = np.histogram(label_im,num_objects)
    mask = label_im==b[h==max(h[1:-1])]
    pylab.imshow(pyramid.ImagePyramid(mask).get_montage(), figure=pylab.figure(figsize=(16,16)))
    mask_volume = np.prod(fm_pixdim) * max(h[1:-1]) / 1000.0
    mask_sm = ndimage.gaussian_filter(mask.astype(np.float), filter_size)
    pylab.imshow(pyramid.ImagePyramid(mask_sm).get_montage(), figure=pylab.figure(figsize=(16,16)))

    fm_Hz = fm.get_data().astype(np.float).squeeze()
    pylab.imshow(pyramid.ImagePyramid(fm_Hz).get_montage(), figure=pylab.figure(figsize=(16,16)))

    fm_Hz_sm = ndimage.gaussian_filter(fm_Hz * mask_sm, filter_size/2)

    pylab.imshow(pyramid.ImagePyramid(fm_Hz_sm).get_montage(), figure=pylab.figure(figsize=(16,16)))

