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

disp("LOADING UTILITIES...")

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
disp("LOADING GLOBAL VARIABLES...");

global DB_MEAN;
global DB_DEVIATION;
global DB_EIGENFACESMATRIX;

global DB_IDENTITYMATRIX;
global DB_DESCRIPTORMATRIX;

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

disp("LOADING FUNCTIONS...")

////
// TEST FUNCTIONS
////

function test_functions()
    global DB_EIGENFACESMATRIX;
    global DB_DESCRIPTORMATRIX;
    global ALREADY_COMPUTED;
    if ALREADY_COMPUTED == 0 then
        eigen_bulk_learning();
        ALREADY_COMPUTED = 1;
    else
        disp("ALREADY COMPUTED !")
    end
    // We show the 12 first eigenfaces
    //disp("SHOWING THE FIRST EIGENFACES :")
    //image = [];
    //for i=1:12
    //    vignette=(matrix(DB_EIGENFACESMATRIX(:,i),[46,56])'*1000+128);
    //    image = [image vignette];
    //end
    //myShowImage(image)
    //pause
    //disp("SHOWING THE FIRST 2 DIMENSION DESCRIPTOR MATRIX :")
    // We show the first two eigenfaces as bases for our descriptors
    //for fo=1:G_TRAINING_FOLDERS
    //    index=((fo-1)*G_TRAINING_FILES);
    //    disp(index)
        //plot2d(DB_DESCRIPTORMATRIX(index+1:index+10,1),DB_DESCRIPTORMATRIX(index+1:index+10,2),style=[color(round(rand()*255),round(rand()*255),round(rand()*255))]);
    //    plot(DB_DESCRIPTORMATRIX(index+1:index+G_TRAINING_FILES,1),DB_DESCRIPTORMATRIX(index+1:index+G_TRAINING_FILES,2),'Color',[rand() rand() rand()], 'linest', 'none', 'markst', '.');
    //end
    //plot(descriptorMatrix(:,1),descriptorMatrix(:,2),'r+');
    //pause   
    // We load an unknown image and get his descriptor
    descriptor=eigen_process("C:\Projects\imr-eigenfaces\dataset\s1\7.pgm");
endfunction

////
// MAIN FUNCTIONS
////


function eigenfaces_main()
endfunction

function eigen_process(path)
    disp("LAUNCHING THE UNIT PROCESS...")
    image=loading_face(path);
    imageNormalized=normalizing_face(image);
    descriptor=descriptor_gen(imageNormalized);
    scoreVector=descriptor_comparison(descriptor);
    person_recognition(scoreVector);
    disp("UNIT DONE !")
endfunction

function eigen_bulk_learning()
    disp("LAUNCHING THE LEARNING PROCESS...")
    // The whole process of learning and creating eigenfaces
    [imageArray, identityArray]=loading_training_faces();
    [m,s]=bulk_image_pre_norm(imageArray);
    Tnorm=bulk_image_norm(imageArray, m, s);
    covariance=bulk_covariance(Tnorm);
    [eigenMatrix, diagMatrix]=bulk_SVD(covariance);
    eigenFaces=store_eigenfaces(eigenMatrix);
    descriptorMatrix=bulk_descriptor_gen(Tnorm,eigenFaces);$
    disp("LEARNING DONE !")
endfunction
////
// The Training Engine
// This is here i put all the components i will use for learning process
////

//// LOADING IMAGES
//TODO
function [T,identityMatrix]=loading_training_faces()
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
            identityMatrix(index)=fo;
        end
    end
    disp("DONE !");
endfunction
//// MEAN AND DEVIATION CALCULUS

function [myMean, myDeviation]=bulk_image_pre_norm(T)
    disp("STARTING GETTING NORMALIZATION ATTRIBUTES...");
    global DB_MEAN;
    global DB_DEVIATION;
    myMean = mean(T,1);
    myDeviation = stdev(T,1);
    
    DB_MEAN = myMean;
    disp("MEAN SAVED TO DB_MEAN !");
    DB_DEVIATION = myDeviation;
    disp("DEVIATION SAVED TO DB_DEVIATION !");
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
    disp("DONE !");
endfunction

//// VECTOR RETRIEVAL

function eigenFaces=store_eigenfaces(eigenMatrix)
    disp("STORING EIGENFACES...");
    global DB_EIGENFACESMATRIX;
    // We retrieve the 48 first columns
    eigenFaces=eigenMatrix(:,1:48);
    DB_EIGENFACESMATRIX=eigenFaces;
    // And that's all !
    disp("EIGENFACES SAVED TO DB_EIGENFACESMATRIX !");
    disp("DONE");
endfunction

//// DESCRIPTOR GENERATION

function descriptorMatrix=bulk_descriptor_gen(Tnorm,eigenFaces)
    disp("COMPUTING DESCRIPTORS...");
    global DB_DESCRIPTORMATRIX;
    descriptorMatrix=Tnorm*eigenFaces;
    DB_DESCRIPTORMATRIX = descriptorMatrix;
    disp("DESCRIPTORS SAVED TO DB_DESCRIPTORS !");
    disp("DONE");
endfunction

////
// The Recognition Engine
// After learning the faces, it's time to recognize them !
////

//// LOADING THE IMAGE TO RECOGNIZE
function T=loading_face(path)
    disp("LOADING THE FACE...");
    // A more simple version of the bulk import
    imageMatrix = myLoadImage(path, 0), // The image must be gray too
    T=matrix(((imresize(imageMatrix, [56 46]))'), 1, 56*46);
    disp("DONE");
endfunction
//TODO

//// NORMALIZING IMAGE
function normalizedImage=normalizing_face(T)
    disp("NORMALIZING THE FACE...");
    // A more simpler version of the bulk normalization
    // No need to replicate the matrix as they are all the same size
    // Acessing the global variables
    global DB_MEAN;
    global DB_DEVIATION;
    normalizedImage = (T-DB_MEAN)./DB_DEVIATION;
    disp("DONE");
endfunction
//TODO

//// DESCRIPTOR GENERATION
function descriptor=descriptor_gen(normalizedImage)
    disp("COMPUTING FACE DESCRIPTOR...");
    global DB_EIGENFACESMATRIX;
    descriptor = normalizedImage*DB_EIGENFACESMATRIX;
    disp("DONE !");
endfunction
//TODO

//// DESCRIPTOR COMPARISON
// That part is pretty complex
// Because it's here we have to interpret the data
// In order to have a pencentage the resemblance of the face
function scoreVector=descriptor_comparison(descriptor)
    disp("GENERATING DESCRIPTOR DISTANCES...");
    global DB_DESCRIPTORMATRIX;
    // I will calculate the distance vector of each descriptions vectors against
    // the description input
    distanceVectors = DB_DESCRIPTORMATRIX - repmat(descriptor, G_TRAINING_FOLDERS*G_TRAINING_FILES, 1);
    // When i got this, i make an euclidian norm of all theses distances vectors
    // using a little math trick
    scoreMatrix = distanceVectors * distanceVectors';
    scoreVector = diag(scoreMatrix);
    // I got now a score : 0 when it's closer, infinite when it's not
    // Indexed for every images of the database
    // I send the Array of output to the final stage
    disp("DONE !");
endfunction

//// PERSON RECOGNITION
function resultArray=person_recognition(scoreVector)
    disp("READING SCORES ...");
    resultArray=[];
    // We'll take the max value (the most far)
    maxValue=max(scoreVector);
    // And the min value (mostly for his index)
    [minValue, index]=min(scoreVector);
    // We make a loop
    for i=1:G_TRAINING_FOLDERS*G_TRAINING_FILES
        // The resultArray contain
        // (1) The index
        resultArray(1,i)=i;
        // (2) The folder index
        resultArray(2,i)=int(i/G_TRAINING_FILES)+1;
        // (3) The score
        resultArray(3,i)=scoreVector(i);
        // (4) The percentage
        resultArray(4,i)=abs((scoreVector(i)/maxValue)-1)*100;
    end
    // Getting the closest
    disp(index);
    folderNumber = int((index-1)/G_TRAINING_FILES)+1;
    fileNumber = modulo((index-1),G_TRAINING_FILES)+1;
    // And we show the results
    disp("THE FACE THAT YOU GAVE REALLY LOOK CLOSE TO THE FOLDER NUMBER "+string(folderNumber)+" (image "+string(fileNumber)+" ).")
endfunction

//// TEST
//// FUNCTIONS
disp("STARTING DEVEL TEST ROUTINE...")
test_functions();
