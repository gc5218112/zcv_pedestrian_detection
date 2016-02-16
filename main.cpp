// HOG_SVM_pedestrians_opencv300.cpp : 定义控制台应用程序的入口点。
// 在Caltech转化过的Inria数据集上，用hog+svm进行训练和测试，实现行人检测
// 包含了hard example的处理

#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <string.h>
#include <sstream>
#include <time.h>
#include <fstream>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector);
void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData);
void compute_hog(const vector< Mat > & img_lst, vector< Mat > & gradient_lst, const Size & size);
void train_svm(const vector< Mat > & gradient_lst, const vector< int > & labels);

void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector){
	// get the support vectors
	Mat sv = svm->getSupportVectors();
	const int sv_total = sv.rows;
	// get the decision function
	Mat alpha, svidx;
	double rho = svm->getDecisionFunction(0, alpha, svidx);

	CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
	CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
		(alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
	CV_Assert(sv.type() == CV_32F);
	hog_detector.clear();

	hog_detector.resize(sv.cols + 1);
	memcpy(&hog_detector[0], sv.ptr(), sv.cols*sizeof(hog_detector[0]));
	hog_detector[sv.cols] = (float)-rho;
}


/*
* Convert training/testing set to be used by OpenCV Machine Learning algorithms.
* TrainData is a matrix of size (#samples x max(#cols,#rows) per samples), in 32FC1.
* Transposition of samples are made if needed.
*/
void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData)
{
	//--Convert data
	const int rows = (int)train_samples.size();
	const int cols = (int)std::max(train_samples[0].cols, train_samples[0].rows);
	cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
	trainData = cv::Mat(rows, cols, CV_32FC1);
	vector< Mat >::const_iterator itr = train_samples.begin();
	vector< Mat >::const_iterator end = train_samples.end();
	for (int i = 0; itr != end; ++itr, ++i)
	{
		CV_Assert(itr->cols == 1 ||
			itr->rows == 1);
		if (itr->cols == 1)
		{
			transpose(*(itr), tmp);
			tmp.copyTo(trainData.row(i));
		}
		else if (itr->rows == 1)
		{
			itr->copyTo(trainData.row(i));
		}
	}
}

/*
读取正训练图片和标注信息，剪裁出正样本窗口区域，并resize到64*128大小
*/
void load_pos_windows(const string& prefix, const string& filename, vector<Mat>& pos_lst, const string& annotation_dir, const Size& mySize){
	ifstream pos_lst_file(prefix + filename);
	if (!pos_lst_file.is_open()){
		cerr << "Unable to open the list of images from " << filename << " filename." << endl;
		exit(-1);
	}
	while (true){
		string line;
		pos_lst_file >> line;
		if (line.empty()){ // no more file to read
			break;
		}
		Mat img = imread((prefix + line).c_str()); // load the image
		if (img.empty()) // invalid image, just skip it.
			continue;

		string annotation_filename = line.substr(4, line.length() - 7) + "txt";
		//cout << "annotation filename is " << annotation_dir + annotation_filename << endl;
		ifstream annotation_file(annotation_dir + annotation_filename);
		int line_cnt = 0;
		string annotation; 
		while (true){
			getline(annotation_file, annotation);
			line_cnt++;
			if (annotation.empty()){
				break;
			}
			//cout << "读取到的annotation 是 " << annotation << "!" << endl;
			if (line_cnt == 1){
				continue;
			}
			string category;
			int x, y, width, height; //data[0]:x  data[1]:y  data[2]:width  data[3]:height
			stringstream dataStream(annotation);
			dataStream >> category; //"person"
			dataStream >> x >> y >> width >> height;
			if (x < 0){
				x = 0;
			}
			if (y < 0){
				y = 0;
			}
			if (x + width > img.cols){
				width = img.cols - x;
			}
			if (y + height > img.rows){
				width = img.rows - y;
			}
			string lineTail;
			dataStream >> lineTail;
			//for (int i = 0; i < 4; i++){
			//	cout << data[i] << ", ";
			//}
			Mat pos_window = img(Rect(x, y, width, height));
			resize(pos_window, pos_window, mySize, 0, 0, CV_INTER_LINEAR);
			pos_lst.push_back(pos_window.clone()); //真心不知道clone有啥效果
		}
		//cout << "本annotation file处理完毕" << endl << endl;
	}
}


/*
载入负训练图片，从中截取64*128大小窗口作为负训练样本
*/
void load_neg_windows(const string& prefix, const string& filename, vector<Mat>& neg_lst, const Size& mySize){
	ifstream file(prefix + filename);
	if (!file.is_open()){
		cerr << "Unable to open the list of images from " << filename << " filename." << endl;
		exit(-1);
	}

	Rect box;
	box.width = mySize.width;
	box.height = mySize.height;

	srand((unsigned int)time(NULL));

	while (true){
		string line;
		getline(file, line);
		if (line.empty()){ // no more file to read
			break;
		}
		Mat img = imread((prefix + line).c_str()); // load the image
		if (img.empty()){ // invalid image, just skip it.
			continue;
		}
		for (int i = 0; i < 10; i++){
			/*
			int size_x = rand() % img.cols + 64;
			int size_y = rand() % img.rows + 128;
			*/
			box.x = rand() % (img.cols - box.width);
			box.y = rand() % (img.rows - box.height);
			/*
			box.width = size_x;
			box.height = size_y;
			if (box.x + box.width > img.cols){
				box.width = img.cols - box.x;
			}
			if (box.y + box.height > img.rows){
				box.height = img.rows - box.y;
			}
			*/
			Mat roi = img(box);
			resize(roi, roi, mySize, 0, 0, CV_INTER_LINEAR);
			neg_lst.push_back(roi.clone());
		}
	}
}

void compute_hog(const vector< Mat > & img_lst, vector< Mat > & gradient_lst, const Size & size){
	HOGDescriptor hog;
	hog.winSize = size;
	Mat gray;
	vector< Point > location;
	vector< float > descriptors;

	vector< Mat >::const_iterator img = img_lst.begin();
	vector< Mat >::const_iterator end = img_lst.end();
	for (; img != end; ++img){
		cvtColor(*img, gray, COLOR_BGR2GRAY);
		hog.compute(gray, descriptors, Size(8, 8), Size(0, 0), location);
		Mat tmpImg = Mat(descriptors).clone();
		//gradient_lst.push_back(Mat(descriptors).clone());
		gradient_lst.push_back(tmpImg);
	}
}

void train_svm(const vector< Mat > & gradient_lst, const vector< int > & labels){
	Mat train_data;
	convert_to_ml(gradient_lst, train_data);

	clog << "Start training...";
	Ptr<SVM> svm = SVM::create();
	/* Default values to train SVM */
	svm->setCoef0(0.0);
	svm->setDegree(3);
	svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-3));
	svm->setGamma(0);
	svm->setKernel(SVM::LINEAR);
	svm->setNu(0.5);
	svm->setP(0.1); // for EPSILON_SVR, epsilon in loss function?
	svm->setC(0.01); // From paper, soft classifier
	svm->setType(SVM::EPS_SVR); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
	svm->train(train_data, ROW_SAMPLE, Mat(labels));
	clog << "...[done]" << endl;

	svm->save("my_people_detector.yml");
}

void test_and_evaluate(const Size & size){
	cout << "Begin testing ..." << endl;
	string inria_test_data_dir = "E:/data-USA/work/data/Inria/test/";
	ifstream finPos(inria_test_data_dir + "pos.lst");//正样本图片的文件名列表
	ofstream foutPos(inria_test_data_dir + "V000.txt");
	string ImgName;
	vector<Rect> locations;
	vector<double> foundWeights;
	HOGDescriptor my_hog;
	vector< float > hog_detector;
	Ptr<SVM> svm;
	my_hog.winSize = size;
	svm = StatModel::load<SVM>("my_people_detector.yml");
	get_svm_detector(svm, hog_detector);
	my_hog.setSVMDetector(hog_detector);

	for (int num = 0; num < 288 && getline(finPos, ImgName); num++){
		//cout << "处理：" << ImgName << endl;
		ImgName = inria_test_data_dir + ImgName;//加上正样本的路径名
		Mat img = imread(ImgName);//读取图片

		locations.clear();
		foundWeights.clear();
		//进行检测
		my_hog.detectMultiScale(img, locations, foundWeights);
		/*
		CV_WRAP virtual void detectMultiScale(const Mat& img, CV_OUT vector<Rect>& foundLocations,
		CV_OUT vector<double>& foundWeights, double hitThreshold = 0,
		Size winStride = Size(), Size padding = Size(), double scale = 1.05,
		double finalThreshold = 2.0, bool useMeanshiftGrouping = false) const;
		*/
		//defaultHog.detectMultiScale(img, found, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 2);
		//

		//画长方形，框出行人
		for (int i = 0; i < locations.size(); i++){
			Rect r = locations[i];
			foutPos << (num + 1) << "," << r.x << "," << r.y << "," << r.width << "," << r.height << "," << foundWeights[i] << endl;
		}
		cout << ".";
	}
	cout << "\n===\nTesting Done.";

	finPos.close();
	foutPos.close();
}

void get_hard_examples(const vector<Mat>& neg_lst, vector<Mat>& hard_lst, const Size & mySize){
	vector<Rect> locations;
	vector<double> foundWeights;
	HOGDescriptor my_hog;
	vector< float > hog_detector;
	Ptr<SVM> svm;
	my_hog.winSize = mySize;
	svm = StatModel::load<SVM>("my_people_detector.yml");
	get_svm_detector(svm, hog_detector);
	my_hog.setSVMDetector(hog_detector);

	for (int i = 0; i < neg_lst.size(); i++){
		Mat img = neg_lst[i];//读取图片

		locations.clear();
		foundWeights.clear();
		//进行检测
		my_hog.detectMultiScale(img, locations, foundWeights);

		//画长方形，框出行人
		for (int i = 0; i < locations.size(); i++){
			Rect r = locations[i];
			Mat hard_example = img(locations[i]);
			resize(hard_example, hard_example, mySize, 0, 0, CV_INTER_LINEAR);
			hard_lst.push_back(hard_example);
		}
	}
}

int do_train(){
	vector<Mat> pos_lst;
	vector<Mat> full_neg_lst;
	vector<Mat> neg_lst;
	vector<Mat> hard_lst;
	vector<Mat> gradient_lst;
	vector<int> labels;
	string train_path_prefix = "E:/data-USA/work/data/Inria/train/";
	string pos = "pos.lst";
	string neg = "neg.lst";
	string annotation_dir = "E:/data-USA/work/data/Inria/train/posGt/";
	Size mySize(64, 128);

	//准备positive training data
	cout << "Loading Positive Training Samples..." << endl;
	load_pos_windows(train_path_prefix, pos, pos_lst, annotation_dir, mySize);
	cout << "\n===\nLoad Positive Training Samples Done." << endl;
	labels.assign(pos_lst.size(), +1);
	const unsigned int old = (unsigned int)labels.size();

	//准备negative training data
	//load_images(neg_dir, neg, full_neg_lst);
	load_neg_windows(train_path_prefix, neg, neg_lst, mySize);
	cout << "\n===\nLoad Negative Images Done." << endl;
	cout << "#(neg in total) = " << neg_lst.size() << endl;
	labels.insert(labels.end(), neg_lst.size(), -1);
	
	CV_Assert(old < labels.size());
	cout << "\n===\ninsert Done." << endl;

	compute_hog(pos_lst, gradient_lst, mySize);
	cout << "\n===\nCompute Positive hog Done." << endl;
	compute_hog(neg_lst, gradient_lst, mySize);
	cout << "\n===\nCompute Negative hog Done." << endl;

	//第一次训练
	train_svm(gradient_lst, labels);
	cout << "\n===\nFirst Train svm Done." << endl;

	//获取false positive samples
	get_hard_examples(neg_lst, hard_lst, mySize);
	labels.insert(labels.end(), hard_lst.size(), -1);

	//计算hard examples的梯度
	compute_hog(hard_lst, gradient_lst, mySize);
	cout << "\n===\nCompute Hard hog Done." << endl;

	//第二次训练
	train_svm(gradient_lst, labels);
	cout << "\n===\nSecond Train svm Done." << endl;

	return 0;

}

int main(){
	//do_train();
	test_and_evaluate(Size(64, 128));
	system("pause");
	return 0;
}