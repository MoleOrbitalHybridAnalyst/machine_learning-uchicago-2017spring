for x in range(0,im.size[0]-imagesize,5):
    for y in range(0,im.size[1]-imagesize,5):
        if cascade.predict(Iimage(pix,x,y,x+imagesize,y+imagesize,1))>0:
            facepatches.append([x,y])
            draw.line([(x,y),(x+imagesize,y),(x+imagesize,y+imagesize),(x,y+imagesize),(x,y)],width=1,fill='white')
    im.save("class_det.png","PNG")
    im.show()

