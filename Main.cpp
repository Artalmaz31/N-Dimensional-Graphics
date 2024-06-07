#include <random>
#include <SFML/Graphics.hpp>
#include <iostream>

int main()
{
	const int maxdim = 6;
	const float PI = 3.1415926;

	int w = 1920;
	int h = 1080;

	int mouseX = w / 2;
	int mouseY = h / 2;

	float mouseSensitivity = 10.0f;
	float speed = 10.0f;
	float mouseWheelSensitivity = 0.01f;

	bool mouseHidden = true;
	bool wasdUD[6] = { false, false, false, false, false, false };

	float dimViewAngles[maxdim - 3] = { 0 };
	float lastDimSwitch = 0;
	int currentDimension = 0;

	int frames = 0;
	float deltaTime = 0;

	sf::Vector3f pos = sf::Vector3f(0.0f, 0.0f, 0.0f);
	sf::Clock clock;

	sf::RenderWindow window(sf::VideoMode(w, h), "N-Dimensional Graphics", sf::Style::Titlebar | sf::Style::Close);
	window.setFramerateLimit(60);
	window.setMouseCursorVisible(false);

	sf::RenderTexture firstTexture;
	firstTexture.create(w, h);
	sf::Sprite firstTextureSprite = sf::Sprite(firstTexture.getTexture());
	sf::Sprite firstTextureSpriteFlipped = sf::Sprite(firstTexture.getTexture());
	firstTextureSpriteFlipped.setScale(1, -1);
	firstTextureSpriteFlipped.setPosition(0, h);

	sf::RenderTexture outputTexture;
	outputTexture.create(w, h);
	sf::Sprite outputTextureSprite = sf::Sprite(outputTexture.getTexture());
	sf::Sprite outputTextureSpriteFlipped = sf::Sprite(firstTexture.getTexture());
	outputTextureSpriteFlipped.setScale(1, -1);
	outputTextureSpriteFlipped.setPosition(0, h);

	sf::Shader shader;
	shader.loadFromFile("Shader.frag", sf::Shader::Fragment);
	shader.setUniform("u_resolution", sf::Vector2f(w, h));

	std::random_device rd;
	std::mt19937 e2(rd());
	std::uniform_real_distribution<> dist(0.0f, 1.0f);

	while (window.isOpen())
	{
		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
			{
				window.close();
			}
			else if (event.type == sf::Event::MouseMoved)
			{
				if (mouseHidden)
				{
					int mx = event.mouseMove.x - w / 2;
					int my = event.mouseMove.y - h / 2;
					mouseX += mx;
					mouseY += my;
					sf::Mouse::setPosition(sf::Vector2i(w / 2, h / 2), window);
				}
			}
			else if (event.type == sf::Event::MouseButtonPressed)
			{
				window.setMouseCursorVisible(false);
				mouseHidden = true;
			}
			else if (event.type == sf::Event::KeyPressed)
			{
				if (event.key.code == sf::Keyboard::Escape)
				{
					window.setMouseCursorVisible(true);
					mouseHidden = false;
				}
				else if (event.key.code == sf::Keyboard::W) wasdUD[0] = true;
				else if (event.key.code == sf::Keyboard::A) wasdUD[1] = true;
				else if (event.key.code == sf::Keyboard::S) wasdUD[2] = true;
				else if (event.key.code == sf::Keyboard::D) wasdUD[3] = true;
				else if (event.key.code == sf::Keyboard::Space) wasdUD[4] = true;
				else if (event.key.code == sf::Keyboard::C) wasdUD[5] = true;

				else if (event.key.code == sf::Keyboard::Tilde)
				{
					for (int i = 0; i < maxdim - 3; i++) { dimViewAngles[i] = 0; }
					currentDimension = 0;
					lastDimSwitch = clock.getElapsedTime().asSeconds();
				}
				else if (clock.getElapsedTime().asSeconds() - lastDimSwitch > 1)
				{
					bool dimension_changed = true;

					if (event.key.code == sf::Keyboard::Add) currentDimension++;
					else if (event.key.code == sf::Keyboard::Subtract) currentDimension--;
					else dimension_changed = false;

					if (currentDimension > maxdim - 4) currentDimension = 0;
					else if (currentDimension < 0) currentDimension = maxdim - 4;
					if (dimension_changed) lastDimSwitch = clock.getElapsedTime().asSeconds();
				}
			}
			else if (event.type == sf::Event::KeyReleased)
			{
				if (event.key.code == sf::Keyboard::W) wasdUD[0] = false;
				else if (event.key.code == sf::Keyboard::A) wasdUD[1] = false;
				else if (event.key.code == sf::Keyboard::S) wasdUD[2] = false;
				else if (event.key.code == sf::Keyboard::D) wasdUD[3] = false;
				else if (event.key.code == sf::Keyboard::Space) wasdUD[4] = false;
				else if (event.key.code == sf::Keyboard::C) wasdUD[5] = false;
			}
			else if (event.type == sf::Event::MouseWheelMoved)
			{
				dimViewAngles[currentDimension] += event.mouseWheel.delta * mouseWheelSensitivity;
			}
		}
		if (mouseHidden)
		{
			float mx = ((float)mouseX / w - 0.5f) * mouseSensitivity;
			float my = ((float)mouseY / h - 0.5f) * mouseSensitivity;

			sf::Vector3f dir = sf::Vector3f(0.0f, 0.0f, 0.0f);
			sf::Vector3f dirTemp;

			if (wasdUD[0]) dir = sf::Vector3f(1.0f, 0.0f, 0.0f);
			else if (wasdUD[2]) dir = sf::Vector3f(-1.0f, 0.0f, 0.0f);
			if (wasdUD[1]) dir += sf::Vector3f(0.0f, -1.0f, 0.0f);
			else if (wasdUD[3]) dir += sf::Vector3f(0.0f, 1.0f, 0.0f);

			dirTemp.z = dir.z * cos(-my) - dir.x * sin(-my);
			dirTemp.x = dir.z * sin(-my) + dir.x * cos(-my);
			dirTemp.y = dir.y;

			dir.x = dirTemp.x * cos(mx) - dirTemp.y * sin(mx);
			dir.y = dirTemp.x * sin(mx) + dirTemp.y * cos(mx);
			dir.z = dirTemp.z;

			pos += dir * speed * (clock.getElapsedTime().asSeconds() - deltaTime);

			if (wasdUD[4]) pos.z -= speed * (clock.getElapsedTime().asSeconds() - deltaTime);
			else if (wasdUD[5]) pos.z += speed * (clock.getElapsedTime().asSeconds() - deltaTime);

			deltaTime = clock.getElapsedTime().asSeconds();

			shader.setUniform("u_mouse", sf::Vector2f(mx, my));
			shader.setUniform("u_time", clock.getElapsedTime().asSeconds());

			shader.setUniform("u_pos", pos);
			shader.setUniformArray("u_dir", dimViewAngles, sizeof(dimViewAngles) / sizeof(dimViewAngles[0]));

			shader.setUniform("u_seed1", sf::Vector2f((float)dist(e2), (float)dist(e2)) * 999.0f);
			shader.setUniform("u_seed2", sf::Vector2f((float)dist(e2), (float)dist(e2)) * 999.0f);
		}
		if (frames % 2 == 1)
		{
			outputTexture.draw(firstTextureSpriteFlipped, &shader);
			window.draw(outputTextureSprite);
		}
		else
		{
			firstTexture.draw(outputTextureSpriteFlipped, &shader);
			window.draw(firstTextureSprite);
		}
		window.display();
		frames++;
	}
	return 0;
}