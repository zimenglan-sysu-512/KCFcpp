#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "kcftracker.hpp"

#include <dirent.h>

using namespace std;
using namespace cv;

cv::Point pre_pt;
cv::Point cur_pt;
bool clicked = false;


void original_track(KCFTracker& tracker, bool SILENT) {
    // Frame readed
    Mat frame;

    // Tracker results
    Rect result;

    // Path to list.txt
    ifstream listFile;
    string fileName = "images.txt";
    listFile.open(fileName);

    // Read groundtruth for the 1st frame
    ifstream groundtruthFile;
    string groundtruth = "region.txt";
    groundtruthFile.open(groundtruth);
    string firstLine;
    getline(groundtruthFile, firstLine);
    groundtruthFile.close();
    
    istringstream ss(firstLine);

    // Read groundtruth like a dumb
    float x1, y1, x2, y2, x3, y3, x4, y4;
    char ch;
    ss >> x1;
    ss >> ch;
    ss >> y1;
    ss >> ch;
    ss >> x2;
    ss >> ch;
    ss >> y2;
    ss >> ch;
    ss >> x3;
    ss >> ch;
    ss >> y3;
    ss >> ch;
    ss >> x4;
    ss >> ch;
    ss >> y4; 

    // Using min and max of X and Y for groundtruth rectangle
    float xMin =  min(x1, min(x2, min(x3, x4)));
    float yMin =  min(y1, min(y2, min(y3, y4)));
    float width = max(x1, max(x2, max(x3, x4))) - xMin;
    float height = max(y1, max(y2, max(y3, y4))) - yMin;
    
    // Read Images
    ifstream listFramesFile;
    string listFrames = "images.txt";
    listFramesFile.open(listFrames);
    string frameName;

    // Write Results
    ofstream resultsFile;
    string resultsPath = "output.txt";
    resultsFile.open(resultsPath);

    // Frame counter
    int nFrames = 0;

    while ( getline(listFramesFile, frameName) ){
        frameName = frameName;

        // Read each frame from the list
        frame = imread(frameName, IMREAD_COLOR); // CV_LOAD_IMAGE_COLOR);

        // First frame, give the groundtruth to the tracker
        if (nFrames == 0) {
            tracker.init( Rect(xMin, yMin, width, height), frame );
            rectangle( frame, Point( xMin, yMin ), Point( xMin+width, yMin+height), Scalar( 0, 255, 255 ), 1, 8 );
            resultsFile << xMin << "," << yMin << "," << width << "," << height << endl;
        }
        // Update
        else{
            result = tracker.update(frame);
            rectangle( frame, Point( result.x, result.y ), Point( result.x+result.width, result.y+result.height), Scalar( 0, 255, 255 ), 1, 8 );
            resultsFile << result.x << "," << result.y << "," << result.width << "," << result.height << endl;
        }

        nFrames++;

        if (!SILENT){
            imshow("Image", frame);
            waitKey(1);
        }
    }
    resultsFile.close();

    listFile.close();
}


int show_image(cv::Mat frame, int frame_idx, int x1, int y1, 
               int x2, int y2, cv::Scalar color, const int wait_time=1) 
{
    cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color);
    cv::imshow("frame_" + std::to_string(frame_idx), frame);
    char c = (char)waitKey(wait_time); // Press  ESC on keyboard to exit
    if(c == 27) {
        return -1;
    }
    return 0;
}


void on_mouse(int event, int x, int y, int flags, void *ustc) {
    if(event == EVENT_LBUTTONDOWN) {
        clicked = true;
        pre_pt = cv::Point(x, y);
    } else if (event == EVENT_LBUTTONUP) {
        clicked = false;
        cur_pt = cv::Point(x, y);
    } else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON)) {
        if(clicked) {
            cur_pt = cv::Point(x, y);
        }
    }
    if(clicked) {
        if(pre_pt.x > cur_pt.x) {
            std::swap(pre_pt.x, cur_pt.x);
        }
        if(pre_pt.y > cur_pt.y) {
            std::swap(pre_pt.y, cur_pt.y);

        }
    }
}


void capture_first_frame_event(cv::Mat frame, int& x1, int& y1, int& x2, int& y2) {
    // the window_name must be the same as the title in imshow
    // return a bounding box that cover the interesting object
    const cv::Scalar color(0, 175, 175);
    const std::string window_name = "frame_0";
    cv::namedWindow(window_name);
    cv::setMouseCallback(window_name, on_mouse, 0);
    
    while(true) {
        int ret = show_image(frame.clone(), 0, x1, y1, x2, y2, color);
        x1 = pre_pt.x;
        y1 = pre_pt.y;
        x2 = cur_pt.x;
        y2 = cur_pt.y;
        if(ret == -1) {
            break;
        }
    }
}


int online_video_capture(KCFTracker& tracker) {
    /*
        Init video capture
    */
    VideoCapture cap(0); 
    if(!cap.isOpened()) {
        std::cout << "Error opening video from camera" << std::endl;
        return -1;
    }

    // Frame readed
    cv::Mat frame;
    // Tracker results
    cv::Rect result;
    // 
    int frame_idx = 0;
    int x1 = 320, y1 = 320, x2 = 480, y2 = 480;

    cap >> frame;
    if(frame.empty()) {
        return -1;
    }
    capture_first_frame_event(frame, x1, y1, x2, y2);

    x1 = std::max(x1, 0);
    y1 = std::max(y1, 0);
    x2 = std::min(x2, frame.cols);
    y2 = std::min(y2, frame.rows);
    std::cout << "init bbox: " << x1 << ", " << y1 << ", " 
                               << x2 << ", " << y2 << std::endl;

    tracker.init(cv::Rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1), frame );
    if(show_image(frame, frame_idx, x1, y1, x2, y2, 
                  cv::Scalar(0, 215, 215)) == -1) 
    {
        return -1;
    }

    while(true) {
        cap >> frame;
        if(frame.empty()) {
            break;
        }
        frame_idx++;
        result = tracker.update(frame);
        std::cout << "upated bbox: " << result.x << ", " << result.y << ", " 
                  << result.x + result.width << ", " << result.y + result.height << std::endl;
        if(show_image(frame, frame_idx, result.x, result.y, 
                      result.x + result.width, 
                      result.y + result.height, 
                      cv::Scalar(0, 255, 255)) == -1) 
        {
            break;
        }
    }

    return 0;
}


int init_track(KCFTracker& tracker, int argc, char* argv[]) {
    if (argc > 5) return -1;

    bool HOG = true;
    bool FIXEDWINDOW = false;
    bool MULTISCALE = true;
    bool SILENT = true;
    bool LAB = false;

    for(int i = 0; i < argc; i++) {
        if(strcmp(argv[i], "hog") == 0)
            HOG = true;
        if(strcmp(argv[i], "fixed_window") == 0)
            FIXEDWINDOW = true;
        if(strcmp(argv[i], "singlescale") == 0)
            MULTISCALE = false;
        if(strcmp(argv[i], "show") == 0)
            SILENT = false;
        if(strcmp(argv[i], "lab") == 0) {
            LAB = true;
            HOG = true;
        }
        if(strcmp(argv[i], "gray") == 0) {
            HOG = false;
        }
    }
    
    tracker = KCFTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

    return 0;
}


int main(int argc, char* argv[]) {
    /*
        Create KCFTracker object
    */
    KCFTracker tracker;
    init_track(tracker, argc, argv);

    // original_track(tracker, SILENT);

    online_video_capture(tracker);

    return 0;
}
