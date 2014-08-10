/*
ERL

Uncopyable
*/

#pragma once

class Uncopyable {
protected:
	Uncopyable() {}
	virtual ~Uncopyable() {}
private:
	Uncopyable(const Uncopyable &);
	Uncopyable &operator=(const Uncopyable &);
};