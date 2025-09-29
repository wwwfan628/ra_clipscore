mkdir train && tar -xvf ILSVRC2012_img_train.tar -C train

cd train
for f in *.tar; do
    d=$(basename "$f" .tar)
    mkdir "$d"
    tar -xvf "$f" -C "$d"
    rm "$f"   # optional: remove the tarball after extraction to save space
done
