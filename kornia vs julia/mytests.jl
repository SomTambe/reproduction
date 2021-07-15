srcs=[[162,237,1] [220,979,1] [1237,861,1] [1254,301,1.]];

tgts=[[193,209,1] [224,854,1] [1308,867,1] [1359,227,1.]];

src_img = load("source.jpeg")

tgt_img = load("target.jpeg")

src2 = imresize(src_img,ratio=1/8)
tgt2 = imresize(tgt_img,ratio=1/8)
src3 = map(x->Gray(x).val, src2)
tgt3 = map(x->Gray(x).val,tgt2)
function warp_me2(img::AbstractArray{T}, tform) where T
   img = ImageTransformations.box_extrapolation(img, RGB(.0,.0,.0))
   # inds = DiffImages.custom_autorange(img, inv(tform))
   inds = map(x->SVector{length(x),Float64}(x.I),CartesianIndices(img))
   # tgts = H * srcs
   # img = map(x -> ImageTransformations._getindex(img, tform(x)), inds)
   img = map(x->img(inv(tform)(x)...),inds)
   return img
end


h=homography2d(srcs,tgts);
h=DiffImages.homography{Float64}(h);
