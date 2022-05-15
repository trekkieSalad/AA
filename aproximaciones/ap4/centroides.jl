using FileIO
using Images

function getCentroide(image::Array{RGB{Normed{UInt8,8}},2})
    canalGris=Gray.(image);
    matrizBooleana = canalGris .> 0.5;
    labelArray = ImageMorphology.label_components(matrizBooleana);
    tamanos = component_lengths(labelArray);
    conservar = findmax(tamanos);
    matrizBooleana = [in(etiqueta,conservar) && (etiqueta!=0) for etiqueta in labelArray];
    if length(ImageMorphology.component_centroids(labelArray)) > 1
        centroide = ImageMorphology.component_centroids(labelArray)[2];
        x = Int(round(centroide[1]));
        y = Int(round(centroide[2]));
    else
        x = 0;
        y = 0;    
    end;
    return (x,y);    
end

