// Clothmain.cpp
// Compile with: g++ main.cpp -o ClothSimulation -lGLEW -lglfw -lGL -lGLU
// Ensure that GLEW, GLFW, and GLM are properly installed and their include paths are correct.

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <vector>
#include <cmath>

// Window dimensions
const unsigned int WIDTH = 800;
const unsigned int HEIGHT = 600;

// Timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// Camera parameters
glm::vec3 cameraPos   = glm::vec3(0.0f, 10.0f, 30.0f); // Elevated for better view
glm::vec3 cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 cameraDirection = glm::normalize(cameraPos - cameraTarget);
glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
glm::vec3 cameraRight = glm::normalize(glm::cross(up, cameraDirection));
glm::vec3 cameraUp    = glm::cross(cameraDirection, cameraRight);

// Declare global view and projection matrices
glm::mat4 view;
glm::mat4 projection;

// Shader sources
const char* vertexShaderSource = R"(
    #version 330 core
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec3 aNormal;

    out vec3 Normal;
    out vec3 FragPos;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main()
    {
        FragPos = vec3(model * vec4(aPos, 1.0));
        Normal = mat3(transpose(inverse(model))) * aNormal;
        gl_Position = projection * view * vec4(FragPos, 1.0);
    }
)";
    
const char* fragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;

    in vec3 Normal;
    in vec3 FragPos;

    uniform vec3 lightPos;
    uniform vec3 viewPos;
    uniform vec3 lightColor;
    uniform vec3 objectColor;

    void main()
    {
        // Ambient
        float ambientStrength = 0.3;
        vec3 ambient = ambientStrength * lightColor;

        // Diffuse
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * lightColor;

        // Specular
        float specularStrength = 0.5;
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);  
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        vec3 specular = specularStrength * spec * lightColor;  
        
        vec3 result = (ambient + diffuse + specular) * objectColor;
        FragColor = vec4(result, 1.0);
    }
)";

// Function to compile a shader
GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    // Check for compilation errors
    int success;
    char infoLog[1024];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if(!success){
        glGetShaderInfoLog(shader, 1024, NULL, infoLog);
        std::cerr << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    return shader;
}

// Function to create a shader program
GLuint createShaderProgram(const char* vertexSrc, const char* fragmentSrc){
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSrc);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSrc);
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    // Check for linking errors
    int success;
    char infoLog[1024];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if(!success){
        glGetProgramInfoLog(shaderProgram, 1024, NULL, infoLog);
        std::cerr <<"ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    return shaderProgram;
}

// Particle class
struct Particle {
    glm::vec3 position;
    glm::vec3 previous_position;
    glm::vec3 acceleration;
    float mass;
    bool movable;

    Particle(glm::vec3 pos) : position(pos), previous_position(pos), acceleration(0.0f), mass(10.0f), movable(true) {} // Increased mass to 10.0f

    void addForce(const glm::vec3& force){
        acceleration += force / mass;
    }

    void timeStep(float deltaTime){
        if (movable){
            glm::vec3 temp = position;
            position += position - previous_position + acceleration * deltaTime * deltaTime;
            previous_position = temp;
            acceleration = glm::vec3(0.0f);
        }
    }

    void offsetPosition(const glm::vec3& offset){
        if (movable){
            position += offset;
        }
    }

    void makeUnmovable(){
        movable = false;
    }

    void makeMovable(){
        movable = true;
    }
};

// Constraint class
struct Constraint {
    Particle* p1;
    Particle* p2;
    float rest_distance;

    Constraint(Particle* p1, Particle* p2) : p1(p1), p2(p2){
        rest_distance = glm::length(p1->position - p2->position);
    }

    void satisfyConstraint(){
        glm::vec3 delta = p2->position - p1->position;
        float current_distance = glm::length(delta);
        if(current_distance == 0.0f) return; // Prevent division by zero
        float diff = (current_distance - rest_distance) / current_distance;
        glm::vec3 correction = delta * 0.5f * diff;
        p1->offsetPosition(correction);
        p2->offsetPosition(-correction);
    }
};

// Cloth class
class Cloth {
public:
    int num_particles_width;
    int num_particles_height;
    std::vector<Particle> particles;
    std::vector<Constraint> constraints;

    Cloth(float width, float height, int num_width, int num_height) :
        num_particles_width(num_width),
        num_particles_height(num_height) {
        // Create particles
        for(int y=0; y<num_particles_height; y++){
            for(int x=0; x<num_particles_width; x++){
                glm::vec3 pos = glm::vec3(width * (x / (float)num_width) - width/2.0f,
                                          -height * (y / (float)num_height) + height/2.0f,
                                          0.0f);
                particles.emplace_back(pos);
            }
        }

        // Connect constraints (structural and shear)
        for(int y=0; y<num_particles_height; y++){
            for(int x=0; x<num_particles_width; x++){
                if(x < num_particles_width -1)
                    constraints.emplace_back(&particles[y*num_particles_width + x],
                                            &particles[y*num_particles_width + (x+1)]);
                if(y < num_particles_height -1)
                    constraints.emplace_back(&particles[y*num_particles_width + x],
                                            &particles[(y+1)*num_particles_width + x]);
                // Shear constraints
                if(x < num_particles_width -1 && y < num_particles_height -1){
                    constraints.emplace_back(&particles[y*num_particles_width + x],
                                            &particles[(y+1)*num_particles_width + (x+1)]);
                    constraints.emplace_back(&particles[y*num_particles_width + (x+1)],
                                            &particles[(y+1)*num_particles_width + x]);
                }
            }
        }

        // Make top row unmovable
        for(int x=0; x<num_particles_width; x++){
            particles[x].makeUnmovable();
        }
    }

    void addForce(const glm::vec3& direction){
        for(auto& p : particles){
            p.addForce(direction);
        }
    }

    void windForce(const glm::vec3& direction){
        // Simple wind force implementation
        for(int y=0; y<num_particles_height -1; y++){
            for(int x=0; x<num_particles_width -1; x++){
                glm::vec3 normal = glm::normalize(glm::cross(
                    particles[y*num_particles_width + x +1].position - particles[y*num_particles_width + x].position,
                    particles[(y+1)*num_particles_width + x].position - particles[y*num_particles_width + x].position
                ));
                glm::vec3 force = normal * glm::dot(normal, direction);
                particles[y*num_particles_width + x].addForce(force);
                particles[y*num_particles_width + x +1].addForce(force);
                particles[(y+1)*num_particles_width + x].addForce(force);
            }
        }
    }

    void timeStep(float deltaTime){
        for(auto& p : particles){
            p.timeStep(deltaTime);
        }

        // Satisfy constraints multiple times for stability
        for(int i=0; i<20; i++){ // Increased iterations for better constraint satisfaction
            for(auto& c : constraints){
                c.satisfyConstraint();
            }
        }
    }
};

// Function to generate mesh data from cloth
void generateMesh(const Cloth& cloth, std::vector<float>& vertices, std::vector<unsigned int>& indices){
    vertices.clear();
    indices.clear();
    std::vector<glm::vec3> normals(cloth.particles.size(), glm::vec3(0.0f));

    // Generate vertices
    for(const auto& p : cloth.particles){
        vertices.push_back(p.position.x);
        vertices.push_back(p.position.y);
        vertices.push_back(p.position.z);
        // Placeholder for normals
        vertices.push_back(0.0f);
        vertices.push_back(0.0f);
        vertices.push_back(0.0f);
    }

    // Generate indices and calculate normals
    for(int y=0; y<cloth.num_particles_height -1; y++){
        for(int x=0; x<cloth.num_particles_width -1; x++){
            int topLeft = y * cloth.num_particles_width + x;
            int topRight = topLeft +1;
            int bottomLeft = (y+1) * cloth.num_particles_width + x;
            int bottomRight = bottomLeft +1;

            // First triangle
            indices.push_back(topLeft);
            indices.push_back(bottomLeft);
            indices.push_back(topRight);

            // Second triangle
            indices.push_back(topRight);
            indices.push_back(bottomLeft);
            indices.push_back(bottomRight);

            // Calculate normals for first triangle
            glm::vec3 v0 = cloth.particles[topLeft].position;
            glm::vec3 v1 = cloth.particles[bottomLeft].position;
            glm::vec3 v2 = cloth.particles[topRight].position;
            glm::vec3 normal1 = glm::normalize(glm::cross(v1 - v0, v2 - v0));
            normals[topLeft] += normal1;
            normals[bottomLeft] += normal1;
            normals[topRight] += normal1;

            // Calculate normals for second triangle
            v0 = cloth.particles[topRight].position;
            v1 = cloth.particles[bottomLeft].position;
            v2 = cloth.particles[bottomRight].position;
            glm::vec3 normal2 = glm::normalize(glm::cross(v1 - v0, v2 - v0));
            normals[topRight] += normal2;
            normals[bottomLeft] += normal2;
            normals[bottomRight] += normal2;
        }
    }

    // Update normals in vertices
    for(int i=0; i<cloth.particles.size(); i++){
        glm::vec3 norm = glm::normalize(normals[i]);
        vertices[i*6 +3] = norm.x;
        vertices[i*6 +4] = norm.y;
        vertices[i*6 +5] = norm.z;
    }
}

// Function to generate a sphere mesh (latitude-longitude)
void generateSphere(float radius, int sectorCount, int stackCount, std::vector<float>& vertices, std::vector<unsigned int>& indices){
    float x, y, z, xy;                              // vertex position
    float nx, ny, nz, lengthInv = 1.0f / radius;    // normal
    float sectorStep = 2 * M_PI / sectorCount;
    float stackStep = M_PI / stackCount;
    float sectorAngle, stackAngle;

    // Vertices
    for(int i = 0; i <= stackCount; ++i){
        stackAngle = M_PI / 2 - i * stackStep;        // from pi/2 to -pi/2
        xy = radius * cosf(stackAngle);             // r * cos(u)
        z = radius * sinf(stackAngle);              // r * sin(u)

        // add (sectorCount+1) vertices per stack
        // the first and last vertices have same position and normal, but different tex coords
        for(int j = 0; j <= sectorCount; ++j){
            sectorAngle = j * sectorStep;           // from 0 to 2pi

            // vertex position
            x = xy * cosf(sectorAngle);             // r * cos(u) * cos(v)
            y = xy * sinf(sectorAngle);             // r * cos(u) * sin(v)
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);

            // normalized vertex normal
            nx = x * lengthInv;
            ny = y * lengthInv;
            nz = z * lengthInv;
            vertices.push_back(nx);
            vertices.push_back(ny);
            vertices.push_back(nz);
        }
    }

    // Indices
    int k1, k2;
    for(int i = 0; i < stackCount; ++i){
        k1 = i * (sectorCount + 1);     // beginning of current stack
        k2 = k1 + sectorCount + 1;      // beginning of next stack

        for(int j = 0; j < sectorCount; ++j, ++k1, ++k2){
            // 2 triangles per sector excluding first and last stacks
            if(i != 0){
                indices.push_back(k1);
                indices.push_back(k2);
                indices.push_back(k1 + 1);
            }

            if(i != (stackCount-1)){
                indices.push_back(k1 + 1);
                indices.push_back(k2);
                indices.push_back(k2 + 1);
            }
        }
    }
}

// Callback to adjust the viewport when the window size changes
void framebuffer_size_callback(GLFWwindow* window, int width, int height){
    glViewport(0, 0, width, height);
}

// Process input
bool windEnabled = true; // Wind is enabled by default

void processInput(GLFWwindow *window){
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    static bool wPressedLastFrame = false;
    if(glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS){
        if(!wPressedLastFrame){
            windEnabled = !windEnabled;
            std::cout << "Wind " << (windEnabled ? "Enabled" : "Disabled") << std::endl;
            wPressedLastFrame = true;
        }
    }
    else{
        wPressedLastFrame = false;
    }
}

// Variables for user interaction
bool isDragging = false;
Particle* selectedParticle = nullptr;

// Sphere parameters
glm::vec3 spherePos(0.0f, -3.0f, 1.0f); // Adjusted Y and Z to prevent initial overlap
float sphereRadius = 2.0f;

// Movement speed
float sphereSpeed = 10.0f;

// Function to convert screen coordinates to world ray
glm::vec3 getRayFromMouse(double mouseX, double mouseY, const glm::mat4& view, const glm::mat4& projection){
    // Convert to normalized device coordinates
    float x = (2.0f * mouseX) / WIDTH - 1.0f;
    float y = 1.0f - (2.0f * mouseY) / HEIGHT;
    float z = 1.0f;
    glm::vec3 ray_nds = glm::vec3(x, y, z);

    // Clip coordinates
    glm::vec4 ray_clip = glm::vec4(ray_nds, 1.0f);

    // Eye coordinates
    glm::vec4 ray_eye = glm::inverse(projection) * ray_clip;
    ray_eye = glm::vec4(ray_eye.x, ray_eye.y, -1.0f, 0.0f);

    // World coordinates
    glm::vec3 ray_wor = glm::vec3(glm::inverse(view) * ray_eye);
    ray_wor = glm::normalize(ray_wor);

    return ray_wor;
}

// Function to find the nearest particle to the ray within a certain threshold
Particle* pickParticle(Cloth& cloth, glm::vec3 rayOrigin, glm::vec3 rayDir, float threshold = 1.0f){
    Particle* closest = nullptr;
    float minDistance = threshold;

    for(auto& p : cloth.particles){
        glm::vec3 toPoint = p.position - rayOrigin;
        float projection = glm::dot(toPoint, rayDir);
        if(projection < 0) continue; // Particle is behind the camera
        glm::vec3 projectedPoint = rayOrigin + rayDir * projection;
        float distance = glm::length(p.position - projectedPoint);
        if(distance < minDistance){
            minDistance = distance;
            closest = &p;
        }
    }

    return closest;
}

// Mouse button callback
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods){
    if(button == GLFW_MOUSE_BUTTON_LEFT){
        if(action == GLFW_PRESS){
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);

            // Retrieve the Cloth object from the window's user pointer
            Cloth* cloth = reinterpret_cast<Cloth*>(glfwGetWindowUserPointer(window));

            glm::vec3 rayDir = getRayFromMouse(xpos, ypos, view, projection);
            glm::vec3 rayOrigin = cameraPos;

            Particle* picked = pickParticle(*cloth, rayOrigin, rayDir, 1.0f);

            if(picked){
                isDragging = true;
                selectedParticle = picked;
                selectedParticle->makeUnmovable();
            }
        }
        else if(action == GLFW_RELEASE){
            if(selectedParticle){
                selectedParticle->makeMovable(); // Allow the particle to move again
                selectedParticle = nullptr;
            }
            isDragging = false;
        }
    }
}

// Cursor position callback
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos){
    if(isDragging && selectedParticle){
        // Convert mouse position to world ray
        glm::vec3 rayDir = getRayFromMouse(xpos, ypos, view, projection);
        glm::vec3 rayOrigin = cameraPos;

        // Find intersection with a plane at the particle's original z position
        if(rayDir.z == 0.0f) return; // Prevent division by zero
        float t = (selectedParticle->position.z - rayOrigin.z) / rayDir.z;
        if(t < 0) return; // Intersection behind the camera

        glm::vec3 newPos = rayOrigin + rayDir * t;
        selectedParticle->position = newPos;
        selectedParticle->previous_position = newPos; // Reset previous position to prevent high velocity
    }
}

int main(){
    // Initialize GLFW
    if(!glfwInit()){
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    // Set GLFW version to 3.3 and use core profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // For MacOS
#endif

    // Create a window
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Cloth Simulation with Interaction", NULL, NULL);
    if(!window){
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    glewExperimental = GL_TRUE; // Needed for core profile
    if(glewInit() != GLEW_OK){
        std::cerr << "Failed to initialize GLEW\n";
        return -1;
    }

    // Set viewport and callback
    glViewport(0, 0, WIDTH, HEIGHT);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // Enable depth testing
    glEnable(GL_DEPTH_TEST);

    // Compile and link shaders
    GLuint shaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);

    // Initialize cloth with increased number of nodes (100x100)
    Cloth cloth(10.0f, 10.0f, 50, 50); // width, height, num_width, num_height

    // Generate initial cloth mesh data
    std::vector<float> clothVertices;
    std::vector<unsigned int> clothIndices;
    generateMesh(cloth, clothVertices, clothIndices);

    // Generate and bind cloth VAO, VBO, EBO
    GLuint clothVAO, clothVBO, clothEBO;
    glGenVertexArrays(1, &clothVAO);
    glGenBuffers(1, &clothVBO);
    glGenBuffers(1, &clothEBO);

    // Bind cloth VAO
    glBindVertexArray(clothVAO);

    // Bind and set cloth VBO
    glBindBuffer(GL_ARRAY_BUFFER, clothVBO);
    glBufferData(GL_ARRAY_BUFFER, clothVertices.size()*sizeof(float), clothVertices.data(), GL_DYNAMIC_DRAW);

    // Bind and set cloth EBO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, clothEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, clothIndices.size()*sizeof(unsigned int), clothIndices.data(), GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Unbind VAO
    glBindVertexArray(0);

    // Generate sphere mesh data
    std::vector<float> sphereVertices;
    std::vector<unsigned int> sphereIndices;
    generateSphere(sphereRadius, 36, 18, sphereVertices, sphereIndices);

    // Generate and bind sphere VAO, VBO, EBO
    GLuint sphereVAO, sphereVBO, sphereEBO;
    glGenVertexArrays(1, &sphereVAO);
    glGenBuffers(1, &sphereVBO);
    glGenBuffers(1, &sphereEBO);

    // Bind sphere VAO
    glBindVertexArray(sphereVAO);

    // Bind and set sphere VBO
    glBindBuffer(GL_ARRAY_BUFFER, sphereVBO);
    glBufferData(GL_ARRAY_BUFFER, sphereVertices.size()*sizeof(float), sphereVertices.data(), GL_STATIC_DRAW);

    // Bind and set sphere EBO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphereIndices.size()*sizeof(unsigned int), sphereIndices.data(), GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Unbind VAO
    glBindVertexArray(0);

    // Set up camera matrices
    view = glm::lookAt(cameraPos, cameraTarget, cameraUp);
    projection = glm::perspective(glm::radians(45.0f), (float)WIDTH/HEIGHT, 0.1f, 100.0f);

    // Set the Cloth object as user pointer for the window to access in callbacks
    glfwSetWindowUserPointer(window, &cloth);

    // Set mouse callbacks
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);

    // Main render loop
    while(!glfwWindowShouldClose(window)){
        // Calculate deltaTime
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // Input
        processInput(window);

        // Handle sphere movement
        if(glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS){
            spherePos.y += sphereSpeed * deltaTime;
        }
        if(glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS){
            spherePos.y -= sphereSpeed * deltaTime;
        }
        if(glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS){
            spherePos.x -= sphereSpeed * deltaTime;
        }
        if(glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS){
            spherePos.x += sphereSpeed * deltaTime;
        }
        if(glfwGetKey(window, GLFW_KEY_COMMA) == GLFW_PRESS){
            spherePos.z -= sphereSpeed * deltaTime;
        }
        if(glfwGetKey(window, GLFW_KEY_PERIOD) == GLFW_PRESS){
            spherePos.z += sphereSpeed * deltaTime;
        }

        // Physics simulation
        cloth.addForce(glm::vec3(0.0f, -9.81f, 0.0f)); // Gravity
        if(windEnabled){
            cloth.windForce(glm::vec3(2.5f, 0.0f, 0.1f)); // Reduced wind force
        }
        cloth.timeStep(deltaTime);

        // Collision Detection between sphere and cloth particles
        // Apply collision correction before constraint satisfaction
        const float epsilon = 0.001f; // Small offset to prevent z-fighting
        for(auto& p : cloth.particles){
            glm::vec3 dir = p.position - spherePos;
            float distance = glm::length(dir);
            if(distance < sphereRadius + epsilon){
                glm::vec3 correction = glm::normalize(dir) * (sphereRadius + epsilon - distance);
                p.position += correction;
                p.previous_position += correction; // Prevent sticking
            }
        }

        // Satisfy constraints again after collision correction
        for(int i=0; i<20; i++){ // Increased iterations for better constraint satisfaction
            for(auto& c : cloth.constraints){
                c.satisfyConstraint();
            }
            // Re-check collision after constraints
            for(auto& p : cloth.particles){
                glm::vec3 dir = p.position - spherePos;
                float distance = glm::length(dir);
                if(distance < sphereRadius + epsilon){
                    glm::vec3 correction = glm::normalize(dir) * (sphereRadius + epsilon - distance);
                    p.position += correction;
                    p.previous_position += correction; // Prevent sticking
                }
            }
        }

        // Update cloth mesh data
        std::vector<float> clothVertices;
        std::vector<unsigned int> clothIndices;
        generateMesh(cloth, clothVertices, clothIndices);

        // Update cloth VBO with new vertex data
        glBindBuffer(GL_ARRAY_BUFFER, clothVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, clothVertices.size()*sizeof(float), clothVertices.data());
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // Clear the screen
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Use shader program
        glUseProgram(shaderProgram);

        // Set view and projection matrices
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        // Set uniforms common to all objects
        glUniform3f(glGetUniformLocation(shaderProgram, "lightPos"), 10.0f, 10.0f, 10.0f);
        glUniform3fv(glGetUniformLocation(shaderProgram, "viewPos"), 1, glm::value_ptr(cameraPos));
        glUniform3f(glGetUniformLocation(shaderProgram, "lightColor"), 1.0f, 1.0f, 1.0f);
        // Note: objectColor will be set individually for cloth and sphere

        // Render Cloth
        {
            glm::mat4 model = glm::mat4(1.0f);
            glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));

            glUniform3f(glGetUniformLocation(shaderProgram, "objectColor"), 0.6f, 0.6f, 0.6f); // Grey color for cloth

            glBindVertexArray(clothVAO);
            glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(clothIndices.size()), GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        }

        // Render Sphere
        {
            glm::mat4 model = glm::translate(glm::mat4(1.0f), spherePos);
            glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));

            glUniform3f(glGetUniformLocation(shaderProgram, "objectColor"), 1.0f, 0.0f, 0.0f); // Red color for sphere

            glBindVertexArray(sphereVAO);
            glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(sphereIndices.size()), GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        }

        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    glDeleteVertexArrays(1, &clothVAO);
    glDeleteBuffers(1, &clothVBO);
    glDeleteBuffers(1, &clothEBO);

    glDeleteVertexArrays(1, &sphereVAO);
    glDeleteBuffers(1, &sphereVBO);
    glDeleteBuffers(1, &sphereEBO);

    glDeleteProgram(shaderProgram);

    glfwTerminate();
    return 0;
}

