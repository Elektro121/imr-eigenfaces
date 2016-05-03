//////
// Image processing project : Face recognition w/ Eigenfaces !
// Copypasta 2016 
////


////
// Pre-execution Routines
// Because calculus needs memory
////
disp ("UNLOCKING MAX MEMORY LIMIT...")
stacksize('max');
disp("DONE, HAVE FUN BUDDY !")

////
// Utilities
////

function imageMatrix=myLoadImage(path,isRGB)
    if isRGB == 0 then
        imageMatrix=double(imread(path));
    else
        imageMatrix=double(rgb2gray(imread(path)));
    end
endfunction
    
function myShowImage(imageMatrix)
    imshow(uint8(imageMatrix))
endfunction

function myWriteImage(imageMatrix,fileName)
    image=imwrite(imageMatrix, fileName);
endfunction

function deg=myRad2Deg(rad)
    deg=rad*(180/%pi);
endfunction


////
// The "Database"
// Here lie the model used for storing all the knowledge my program can gather 
// on faces
////



////
// MAIN VARIABLES
////

// By default 
//G_TRAINING_PATH="C:\Projects\imr-eigenfaces\dataset\";
//G_TRAINING_FOLDERS=40;
//G_TRAINING_FILES=10;

G_TRAINING_PATH="C:\Projects\imr-eigenfaces\dataset\";
G_TRAINING_FOLDERS=40;
G_TRAINING_FILES=5; // Maximum files is limited to 10 (architectures choices)

////
// TEST FUNCTIONS
////

function test_functions()
    imageArray=loading_training_faces();
    [m,s]=bulk_image_pre_norm(imageArray);
    Tnorm=bulk_image_norm(imageArray, m, s);
    covariance=bulk_covariance(Tnorm);
    [eigenMatrix, diagMatrix]=bulk_SVD(covariance)
    disp(size(eigenMatrix))
    disp(size(diagMatrix))
    image = [];
    for i=1:12
        vignette=(matrix(eigenMatrix(:,i),[46,56])'*1000+128);
        image = [image vignette];
    end
    myShowImage(image)
    
    pause
endfunction

////
// MAIN FUNCTIONS
////


function eigenfaces_main()
endfunction

////
// The Training Engine
// This is here i put all the components i will use for learning process
////

//// LOADING IMAGES
//TODO
function T=loading_training_faces()
    disp("STARTING LOADING TRAINING FACES...");
    imageArray=zeros((G_TRAINING_FOLDERS*G_TRAINING_FILES),56*46);
    for fo=1:G_TRAINING_FOLDERS
        for fi=1:G_TRAINING_FILES
            // This is where i'm forcing my architecture choices :D
            imageLocation = G_TRAINING_PATH + "s" + string(fo) + "\" + string(fi) + ".pgm";
            imageMatrix = myLoadImage(imageLocation, 0), // They're all grey
            // to s1/1.pgm = 11 from sn/m.pgm = (n*10)+ m so varying to 11 from 20 max
            index=((fo-1)*G_TRAINING_FILES)+fi; 
            // But they're not at the right size, which is normalised about 46x56px
            T(index,:)=matrix(((imresize(imageMatrix, [56 46]))'), 1, 56*46);
        end
    end
    disp("DONE !");
endfunction
//// MEAN AND DEVIATION CALCULUS

function [myMean, myDeviation]=bulk_image_pre_norm(T)
    disp("STARTING GETTING NORMALIZATION ATTRIBUTES...");
    myMean = mean(T,1);
    myDeviation = stdev(T,1);
    disp("DONE !");
endfunction

//// IMAGE NORMALIZATION

function Tnorm=bulk_image_norm(T, myMean, myDeviation)
    disp("STARTING NORMALIZATION...");
    Tnorm = (T-repmat(myMean, G_TRAINING_FOLDERS*G_TRAINING_FILES, 1))./repmat(myDeviation, G_TRAINING_FOLDERS*G_TRAINING_FILES, 1 );
    disp("DONE !");
endfunction

//// COVARIANCE

function covariance=bulk_covariance(Tnorm)
    disp("STARTING COVARIANCE PROCESSING...");
    covariance = cov(Tnorm);
    disp("DONE !");
endfunction

//// SVD

function [eigenMatrix,diagMatrix]=bulk_SVD(covariance)
    disp("STARTING SINGLE VALUE DECOMPOSITION...");
    [eigenMatrix, diagMatrix, V] = svd(covariance);
    disp("DONE");
endfunction

//// VECTOR RETRIEVAL
//TODO

//// DESCRIPTOR GENERATION
//TODO

////
// The Recognition Engine
// After learning the faces, it's time to recognize them !
////

//// LOADING THE IMAGE TO RECOGNIZE
//TODO

//// NORMALIZING IMAGE
//TODO

//// DESCRIPTOR GENERATION
//TODO

//// DESCRIPTOR COMPARISON
//TODO

//// PERSON RECOGNITION


//// TEST
//// FUNCTIONS
test_functions();
