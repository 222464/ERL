/*
ERL

Software Image 2D
*/

#pragma

#include <vector>

namespace erl {
	template<class T>
	class SoftwareImage2D {
	private:
		std::vector<T> _data;

		int _width, _height;

	public:
		void reset(int width, int height, T defaultValue = T());

		T getPixel(int x, int y) const {
			return _data[x + y * _width];
		}

		void setPixel(int x, int y, const T &value) {
			_data[x + y * _width] = value;
		}

		T* getData() {
			return &_data[0];
		}

		int getWidth() {
			return _width;
		}

		int getHeight() {
			return _height;
		}
	};

	template<class T>
	void SoftwareImage2D<T>::reset(int width, int height, T defaultValue) {
		_width = width;
		_height = height;

		_data.clear();
		_data.assign(_width * _height, defaultValue);
	}
}