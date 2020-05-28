#include <opencv2/opencv.hpp>
#include <iostream>
#include <array>
#include <vector>
#include <filesystem>

using namespace std;
using namespace std::filesystem;
using namespace cv;

int main() {
	bool is_default;
	while (true) {
		string answer;
		cout << "Use test values (test.png must exist)?: (y/n)\n";
		cin >> answer;
		if (answer == "y") {
			is_default = true;
			break;
		}
		else if (answer == "n") {
			is_default = false;
			break;
		}
	}

	Matx<double, 4, 1> Xp, Yp;
	Mat img;
	double h2w_ratio;

	if (is_default) {
		img = imread("test.png");
		if (img.empty()) {
			throw "Could not read test.png";
		}

		/*Xp = { 299 * 8, 1096 * 8, 1197 * 8, 84 * 8 };
		Yp = { 134 * 8, 57 * 8, 768 * 8, 592 * 8 };*/
		Xp = { 299, 1096, 1197, 84 };
		Yp = { 134, 57, 768, 592 };
		h2w_ratio = 9. / 16;
	}
	else {
		String image_path;
		while (true) {
			cout << "Enter image path:\n";
			cin >> image_path;
			if (exists(image_path)) {
				bool is_desired;
				while (true) {
					cout << "Should to use " << absolute(image_path) << "?: (y/n)\n";
					string answer;
					cin >> answer;
					if (answer == "y") {
						is_desired = true;
						break;
					}
					else if (answer == "n") {
						is_desired = false;
						break;
					}
				}
				if (is_desired) break;
			}
		}
		img = imread(image_path);

		while (true) {
			cout << "Enter height to width ratio: (>0, float)\n";
			cin >> h2w_ratio;
			if (h2w_ratio > 0) break;
		}

		cout << "Enter 4 points: 0<=X<=" << img.cols << " 0<=Y<=" << img.rows << "\n";
		for (uint8_t i = 0; i < 4; ++i) {
			int temp_x, temp_y;
			while (true) {
				cout << "Point " << to_string(i + 1) << ": (X Y)\n";
				cin >> temp_x >> temp_y;
				if (temp_x >= 0 && temp_x < img.cols && temp_y >= 0 && temp_y < img.rows) {
					break;
				}
				else {
					cout << "Wrong value(s)!\n";
				}
			}
			Xp(i, 0) = temp_x;
			Yp(i, 0) = temp_y;
		}
	}

	cout << "Doing things...\n";


	Matx<double, 4, 1> X, Y;

	const double center_x = sum(Xp)[0] / 4;
	const double center_y = sum(Yp)[0] / 4;

	uint8_t r[2][2];
	for (uint8_t i = 0; i < 4; ++i) {
		r[Xp(i, 0) >= center_x][Yp(i, 0) >= center_y] = i;
	}

	double  _width1 = min(abs(Xp(r[0][0]) - Xp(r[1][0])), abs(Xp(r[0][1]) - Xp(r[1][1])));
	double _height1 = min(abs(Yp(r[0][0]) - Yp(r[0][1])), abs(Yp(r[1][0]) - Yp(r[1][1])));
	double width1 = static_cast<int>(max(1., _width1));
	double height1 = static_cast<int>(max(1., min(h2w_ratio * width1, _height1)));

	for (uint8_t i = 0; i < 2; ++i) {
		for (uint8_t j = 0; j < 2; ++j) {
			X(r[i][j]) = i * width1;
			Y(r[i][j]) = j * height1;
		}
	}

	Mat B;
	hconcat(X, Y, B);
	hconcat(B, Mat::ones(4, 1, CV_64FC1), B);
	hconcat(B, Mat::zeros(4, 3, CV_64FC1), B);
	hconcat(B, -Xp.mul(X), B);
	hconcat(B, -Xp.mul(Y), B);
	hconcat(B, Mat::zeros(4, 3, CV_64FC1), B);
	hconcat(B, X, B);
	hconcat(B, Y, B);
	hconcat(B, Mat::ones(4, 1, CV_64FC1), B);
	hconcat(B, -Yp.mul(X), B);
	hconcat(B, -Yp.mul(Y), B);

	B = B.reshape(0, 8);

	Mat D;
	hconcat(Xp, Yp, D);
	D = D.reshape(0, 8);

	Mat l = (B.t() * B).inv() * B.t() * D;

	Mat A;
	hconcat(l(Range(0, 6), Range(0, 1)).t(), Matx<double, 1, 3>(0, 0, 1), A);
	A = A.reshape(0, 3);

	Mat C;
	hconcat(l(Range(6, 8), Range(0, 1)).t(), Matx<double, 1, 1>(1), C);

	Mat img1 = Mat(height1, width1, CV_8UC3);

	int x, y, x1, y1;
	Mat t;

	auto tic = chrono::high_resolution_clock::now();
#pragma omp parallel for collapse(2) shared(img, img1, A, C) private(x, x1, y, y1, t)	
	for (y = 0; y < img1.rows; ++y) {
		for (x = 0; x < img1.cols; ++x) {
			t = A * Matx<double, 3, 1>(x, y, 1) / (C * Matx<double, 3, 1>(x, y, 1));
			x1 = round(t.at<double>(0, 0));
			y1 = round(t.at<double>(1, 0));
			if (x1 >= 0 && y1 >= 0 && x1 < img.cols && y1 < img.rows) {
				img1.at<Vec3b>(Point(x, y)) = img.at<Vec3b>(Point(x1, y1));
			}
		}
	}
	auto toc = chrono::high_resolution_clock::now();

	cout << chrono::duration_cast<chrono::nanoseconds>(toc - tic).count() / 1.e9 << endl;

	// blur(img1, img1, Size(2, 2));
	namedWindow("Result", WINDOW_NORMAL);
	imshow("Result", img1);

	imwrite("res.png", img1);
	cout << "Result saved to res.png.";

	waitKey(0);
	return 0;
}