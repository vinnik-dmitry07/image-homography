#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <mpi.h>

using namespace std;
using namespace std::filesystem;
using namespace chrono;
using namespace cv;

inline int main_mpi(int argc, char* argv[]) {
	int rank, comm_size;
	int width, height, width1, height1;
	double h2w_ratio;
	high_resolution_clock::time_point tick, tock;
	Mat A, C, img, img1;
	Matx<double, 4, 1> Xp, Yp;
	
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	if (rank == 0) {
		String image_path;
		if (argc == 11)
			image_path = argv[1];
		else
			image_path = "tests/test.png";

		if (!exists(image_path)) throw "No such a file!";

		//cout << "Loading image...\n";
		img = imread(image_path);	

		if (img.empty()) throw "Could not read the image";

		if (argc == 11) {
			h2w_ratio = atof(argv[2]);
			Xp = { atof(argv[3]), atof(argv[4]), atof(argv[5]), atof(argv[6]) };
			Yp = { atof(argv[7]), atof(argv[8]), atof(argv[9]), atof(argv[10]) };
		}
		else {
			h2w_ratio = 9. / 16;
			Xp = { 299, 1096, 1197, 84 };
			Yp = { 134, 57, 768, 592 };
		}

		width = img.size().width;
		height = img.size().height;

		//cout << "Doing things...\n";

		Matx<double, 4, 1> X, Y;

		const double center_x = sum(Xp)[0] / 4;
		const double center_y = sum(Yp)[0] / 4;

		uint8_t r[2][2];
		for (uint8_t i = 0; i < 4; ++i) {
			r[Xp(i, 0) >= center_x][Yp(i, 0) >= center_y] = i;
		}

		double  _width1 = min(abs(Xp(r[0][0]) - Xp(r[1][0])), abs(Xp(r[0][1]) - Xp(r[1][1])));
		double _height1 = min(abs(Yp(r[0][0]) - Yp(r[0][1])), abs(Yp(r[1][0]) - Yp(r[1][1])));
		width1 = static_cast<int>(max(1., _width1));
		height1 = static_cast<int>(max(1., min(h2w_ratio * width1, _height1)));

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

		hconcat(l(Range(0, 6), Range(0, 1)).t(), Matx<double, 1, 3>(0, 0, 1), A);
		A = A.reshape(0, 3);

		hconcat(l(Range(6, 8), Range(0, 1)).t(), Matx<double, 1, 1>(1), C);
		
		tick = high_resolution_clock::now();
	}

	MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&width1, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&height1, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (rank != 0) {
		A.create(3, 3, CV_64FC1);
		C.create(1, 3, CV_64FC1);
		img.create(height, width, CV_8UC3);
	}
	MPI_Bcast(A.data, 3 * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(C.data, 1 * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(img.data, height * width * 3, MPI_UINT8_T, 0, MPI_COMM_WORLD);

	auto shift = height1 / comm_size;
	Mat sub_img = Mat(shift, width1, CV_8UC3);

	for (int _y = 0; _y < sub_img.rows; ++_y) {
		for (int x = 0; x < sub_img.cols; ++x) {
			int y = _y + rank * shift;
			Mat temp = A * Matx<double, 3, 1>(x, y, 1) / (C * Matx<double, 3, 1>(x, y, 1));
			int x1 = static_cast<int>(round(temp.at<double>(0, 0)));
			int y1 = static_cast<int>(round(temp.at<double>(1, 0)));
			if (x1 >= 0 && y1 >= 0 && x1 < img.cols && y1 < img.rows) {
				sub_img.at<Vec3b>(Point(x, _y)) = img.at<Vec3b>(Point(x1, y1));
			}
		}
	}

	if (rank == 0) {
		img1 = Mat(height1, width1, CV_8UC3);
	}

	MPI_Gather(sub_img.data, static_cast<int>(sub_img.total()) * 3, MPI_UINT8_T, img1.data, static_cast<int>(sub_img.total()) * 3, MPI_UINT8_T, 0, MPI_COMM_WORLD);
	MPI_Finalize();

	if (rank == 0) {
		tock = high_resolution_clock::now();
		cout << chrono::duration_cast<chrono::nanoseconds>(tock - tick).count() / 1.e9 << endl;
		// blur(img1, img1, Size(2, 2));
		/*namedWindow("Result", WINDOW_NORMAL);
		imshow("Result", img1);

		imwrite("res.png", img1);
		cout << "Result saved to res.png.";

		waitKey(0);*/
	}
	return 0;
}