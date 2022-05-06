
using FileIO
using Images




######################################################################################################################
# Caracteristicas morfologicas de imagenes o partes de imagenes:
# Cargamos la imagen
imagen = load(pwd() * "/ejemplos/gg (246).jpg"); display(imagen);
#imagen = load(pwd() * "/ejemplos/image(150).jpg"); display(imagen);
image=convert(Array{Float64,2}, gray.(Gray.(imagen)))
img_gray = @. Gray(0.8 * Gray.(imagen) > 0.7);
img_morphograd = morpholaplace(img_gray)
#println(gray.(img_morphograd))
display(img_gray);
sleep(2)
#display(img_morphograd);
mayores = image .> 0.5;
menores = image .< 0.5;
#mio = Gray.(normalizeInputs(broadcast(abs, (image .+ (reverse(image, dims=2) .* mayores) .- (1 .- reverse(image, dims=2) .* menores)))));
#mio = Gray.(normalizeInputs(broadcast(abs, (image .+ (image .* mayores) .- (1 .- image .* menores)))));
#mio = Gray.(normalizeInputs(broadcast(abs, (image .- reverse(image, dims=1)))));
#display(mio);
#save(pwd() * "/ejemplos/difno3.jpg", mio);

sleep(2)


# Vamos a detectar los objetos rojos
#  Aquellos cuyo valor de rojo es superior en cierta cantidad al valor de verde y azul
# Definimos en que cantidad queremos que sea mayor
canalGris=Gray.(imagen);
matrizBooleana = canalGris .> 0.5;
#matrizBooleana = gray.(img_morphograd) .> 0.0;
# Mostramos esta matriz booleana para ver que objetos ha encontrado
sleep(2);
display(Gray.(matrizBooleana));


# Aplicamos esta funcion a la matriz booleana (imagen umbralizada) que construimos antes:
labelArray = ImageMorphology.label_components(matrizBooleana);
# Cuantos objetos se han detectado:
println("Se han detectado $(maximum(labelArray)) objetos")

# A partir de aqui se pueden extraer distintas caracteristicas, como pueden las siguientes:
#  Devuelven una caracteristica por cada etiqueta distinta, incluyendo como primera la etiqueta "0"
boundingBoxes = ImageMorphology.component_boxes(labelArray);
tamanos = ImageMorphology.component_lengths(labelArray);
pixeles = ImageMorphology.component_indices(labelArray);
pixeles = ImageMorphology.component_subscripts(labelArray);
centroides = ImageMorphology.component_centroids(labelArray);

# Sin embargo, suele ser util filtrar los objetos en primer lugar y eliminar los muy grandes o muy pequeños
# Calculamos los tamaños
tamanos = component_lengths(labelArray);
# Que etiquetas son de objetos demasiado pequeños (30 pixeles o menos):
etiquetasEliminar = findmax(tamanos); # Importate el -1, porque la primera etiqueta es la 0
# Se construye otra vez la matriz booleana, a partir de la matriz de etiquetas, pero eliminando las etiquetas indicadas
# Para hacer esto, se hace un bucle sencillo en el que se itera por cada etiqueta
#  Esto se realiza de forma sencilla con la siguiente linea
matrizBooleana = [in(etiqueta,etiquetasEliminar) && (etiqueta!=0) for etiqueta in labelArray];
sleep(2);
display(Gray.(matrizBooleana));


# Con esos objetos rojos "grandes", se toman de nuevo las etiquetas
labelArray = ImageMorphology.label_components(matrizBooleana);
# Cuantos objetos se han detectado:
println("Se han detectado $(maximum(labelArray)) objetos rojos grandes")

# Vamos a situar el centroide de estos objetos en la imagen umbralizada, poniéndolo en color rojo
# Por tanto, hay que construir una imagen en color:
imagenObjetos = RGB.(matrizBooleana, matrizBooleana, matrizBooleana);
# Calculamos los centroides, y nos saltamos el primero (el elemento "0"):
centroides = ImageMorphology.component_centroids(labelArray)[2];
# Para cada centroide, ponemos su situacion en color rojo
for centroide in centroides
    println(centroide);
    x = Int(round(centroide[1]));
    y = Int(round(centroide[2]));
    imagenObjetos[ x, y ] = RGB(1,0,0);
end;

# Vamos a recuadrar el bounding box de estos objetos, en color verde
# Calculamos los bounding boxes, y eliminamos el primero (el objeto "0")
boundingBoxes = ImageMorphology.component_boxes(labelArray)[2:end];
for boundingBox in boundingBoxes
    x1 = boundingBox[1][1];
    y1 = boundingBox[1][2];
    x2 = boundingBox[2][1];
    y2 = boundingBox[2][2];
    imagenObjetos[ x1:x2 , y1 ] .= RGB(0,1,0);
    imagenObjetos[ x1:x2 , y2 ] .= RGB(0,1,0);
    imagenObjetos[ x1 , y1:y2 ] .= RGB(0,1,0);
    imagenObjetos[ x2 , y1:y2 ] .= RGB(0,1,0);
end;
sleep(2);
display(imagenObjetos);

# Y hacemos lo mismo con la imagen original:
for boundingBox in boundingBoxes
    x1 = boundingBox[1][1];
    y1 = boundingBox[1][2];
    x2 = boundingBox[2][1];
    y2 = boundingBox[2][2];
    imagen[ x1:x2 , y1 ] .= RGB(0,1,0);
    imagen[ x1:x2 , y2 ] .= RGB(0,1,0);
    imagen[ x1 , y1:y2 ] .= RGB(0,1,0);
    imagen[ x2 , y1:y2 ] .= RGB(0,1,0);
end;
sleep(2);
display(imagen);


# Finalmente, guardamos esta imagen
# save("imagenProcesada.jpg", imagen)
