srcs=[[162,237,1] [220,979,1] [1237,861,1] [1254,301,1.]];

tgts=[[193,209,1] [224,854,1] [1308,867,1] [1359,227,1.]];

src_img = load("/home/somvt/work/DiffImages.jl/examples/source.jpeg")'

tgt_img = load("/home/somvt/work/DiffImages.jl/examples/target.jpeg")'

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
function L1_float(y,ŷ)
    loss = 0
    for i in CartesianIndices(y)
        if checkbounds(Bool,ŷ,i)
            s = ŷ[i]-y[i]
            if s > 0    
                loss += s
            else
                loss += -s
            end
        else
            if y[i] > 0    
                loss += y[i]
            else
                loss += -y[i]
            end
        end
    end
    loss
end

function L1_color(y,ŷ)
    # ŷ => warped image, y => target image
    loss = 0
    for i in CartesianIndices(y)
        if checkbounds(Bool,ŷ,i)
            r = ŷ[i]-y[i]
            loss += abs(r.r+r.g+r.b)
        else
            c = y[i]
            loss += abs(c.r+c.g+c.b)
        end
    end
    loss
end
# L(y,ȳ) = sum(abs.(ȳ-begin
#                    y = map(x->x.val,y)
#                    y
#                    end)) # => y in Gray
# for i in 1:num_iters
#     ∇ = Zygote.gradient(src, h) do img, tform
#         c = sum(warp_me2(img, tform))
#         gray(c)
#         end[2].H
#     h = map((y,ȳ)->map(L(y,ȳ)),m,∇)
#     h = homography(SMatrix{3,3,Float64,9}(h))
# end
