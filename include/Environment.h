#pragma once
#include <stdlib.h>
#include <GL/gl.h>
#include <SDL2/SDL.h>
#include <vector>
#include <string>
#include <iostream>
// #include <chrono>

class Environment
{
protected:
	SDL_Keycode key;
	SDL_Window* window_ = nullptr;
	std::vector<unsigned char> red;
	std::vector<unsigned char> green;
	std::vector<unsigned char> blue;
	std::vector<unsigned char> buffer;
	unsigned int buffer_id;
	int window_width;
	int window_height;
	int buffer_width;
	int buffer_height;
	// std::chrono::time_point<std::chrono::steady_clock> start;
public:
	//Constructor
	Environment(int buf_w = 84, int buf_h = 84, int win_w = 336, int win_h = 336, bool display = false) : buffer_width(buf_w), buffer_height(buf_h), window_width(win_w), window_height(win_h), buffer(buf_w*buf_h), red(buf_w*buf_h), green(buf_w*buf_h), blue(buf_w*buf_h)
	{	
		//Create window
		SDL_Init(SDL_INIT_EVERYTHING);
		window_ = SDL_CreateWindow("Visual",SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,win_w,win_h,SDL_WINDOW_OPENGL | (display ? SDL_WINDOW_SHOWN : SDL_WINDOW_HIDDEN) );
		SDL_GLContext context = SDL_GL_CreateContext(window_);
		SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

		//Enable depth test
		glEnable(GL_DEPTH_TEST);

		//Buffer settings
		glEnable(GL_TEXTURE_2D);
		glGenTextures(1, &buffer_id);
		glBindTexture(GL_TEXTURE_2D, buffer_id);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, buffer_width, buffer_height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
		// glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, buffer_width, buffer_height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
			// glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			// glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		//Switch back to default texture
		glBindTexture(GL_TEXTURE_2D, 0);

		//Record current time
		// start = std::chrono::steady_clock::now();
	}

	//Check keyboard input
	void keyboard()
	{
		key = 0;
		SDL_Event event;
		while(SDL_PollEvent(&event)==true)
		{
			if(event.type == SDL_QUIT)
				exit(0);
			if(event.type == SDL_KEYUP)
				key = event.key.keysym.scancode;
		}
	}

	//Render game to screen
	void render()
	{
		// Default Background color 
		glClearColor(0, 0, 0, 1.0);

		//Set the texture
		glBindTexture(GL_TEXTURE_2D, buffer_id);

		//Capture scene into texture
		glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, buffer_width, buffer_height);

		//Get texture color channels
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_UNSIGNED_BYTE, buffer.data());

		//Clear the rendered scene
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//Set the viewport
		glMatrixMode(GL_PROJECTION); glLoadIdentity();
		glViewport(0, 0, window_width, window_height);
		glMatrixMode(GL_MODELVIEW);  glLoadIdentity();
		
		//Draw projection plane
		glColor3f(1.0f, 1.0f, 1.0f);
		glBegin(GL_QUADS);
			glTexCoord2i(0, 0); glVertex3f(-1, -1, 0);
			glTexCoord2i(0, 1); glVertex3f(-1,  1, 0);
			glTexCoord2i(1, 1); glVertex3f( 1,  1, 0);
			glTexCoord2i(1, 0); glVertex3f( 1, -1, 0);
		glEnd();
	
		//Swap buffers to show
		SDL_GL_SwapWindow(window_);
		
		//Check keyboard
		keyboard();

		//Set to default
		glBindTexture(GL_TEXTURE_2D, 0);
		glMatrixMode(GL_PROJECTION); glLoadIdentity();
		glViewport(0, 0, buffer_width, buffer_height);
		glMatrixMode(GL_MODELVIEW);  glLoadIdentity();
	}

	//Check if the game if over
	virtual bool game_over() {}

	//Reset the game
	virtual void reset_game() {}

	//Draw the game
	virtual void draw() {}

	//Step game (overwrite this)
	virtual float step(std::vector<float> action)
	{
		//Render the scene
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glBegin(GL_TRIANGLES);
				glColor3f(1.0f, 0.0f, 0.0f);
				glVertex2f(-1, -1);
				glVertex2f(-1, 1);
				glVertex2f(0, 1);
			glEnd();

		render();
		return 0;
	}

	//Get the key that was pressed
	SDL_Keycode key_code() {return key;}

	//Get the current score
	virtual int score(){}
	
	//Get the screen
	std::vector<unsigned char> screen(){ return buffer; }

	//Show the window
	void show() { SDL_ShowWindow(window_); }

	//Hide window
	void hide() { SDL_HideWindow(window_); }

	//Get runtime
	double time()
	{
		// auto end = std::chrono::steady_clock::now();
		// std::chrono::duration<double> diff = end-start;
		// return diff.count();
	}

	// void snapshot(string file = "snapshot.png")
	// {
	// 	// SDL_SavePNG(SDL_GetWindowSurface(window_),file.data());
	// }
};