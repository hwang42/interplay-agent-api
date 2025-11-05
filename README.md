# AIC2OR Project - Agent API

Public stateful API for interacting with Astro agents.

> Note: This version of the repository is prepared for public release, please [contact the research team](mailto://hwang99@ur.rochester.edu) for prompt templates (i.e., `*.jinja2` files).

## Setup and Run the Project

This project is containerized with `Docker` and run with `Docker Compose` for easy setup and consistent environments.

The provided `Dockerfile` uses Python 3.12 (slim image) and runs as a non-root user for improved security.

1. **:package: Prerequisites**

   Make sure you have the following installed:

   - [Docker](https://docs.docker.com/engine/install/)
   - [Docker Compose](https://docs.docker.com/compose/install/)

2. **:gear: Configure Environment Variables**

   Before running the project, ensure the required environment variables are set in your `docker-compose.yml` or a separate `.env` file.

   At minimum, you must provide:

   | Variable                                                          |      Required      | Description                                                                                                                                         |
   | :---------------------------------------------------------------- | :----------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------- |
   | `MASTER_KEY`                                                      | :white_check_mark: | API key to access the FastAPI service                                                                                                               |
   | `OPENAI_API_KEY`                                                  | :white_check_mark: | Your OpenAI API key (we respect other `openai` library's environment variables)                                                                     |
   | `SQLALCHEMY_DATABASE_URL`                                         | :white_check_mark: | SQLAlchemy database connection string ([SQLAlchemy database URLs documentation](https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls)) |
   | `LANGFUSE_HOST`<br>`LANGFUSE_SECRET_KEY`<br>`LANGFUSE_PUBLIC_KEY` |                    | Langfuse integration for monitoring and trace logging (we respect other `langfuse` library's environment variables)                                 |
   | `TZ`                                                              |                    | Set the container's timezone (default: UTC)                                                                                                         |

   The provided `docker-compose.yml` file provides a completely self-contained setup.

3. **:rocket: Build and Run the Services**

   Build and start the containers:

   ```bash
   sudo docker compose up -d --build
   ```

   This will:

   - Build the `agent-api` image
   - Start the FastAPI backend on port 5000
   - Start the LangFuse backend on port 3000
   - Automatically restart the containers unless stopped

4. **:computer: Access the API and Documentation**

   Once running, the API will be available at:

   [http://localhost:5000](http://localhost:5000)

   with the documentations available at:

   [http://localhost:5000/docs](http://localhost:5000/docs)

5. **:memo: (Optional) Access the Langfuse Interface**

   If you choose to setup the Langfuse service using the provided `docker-compose.yml`, it will be available at:

   [http://localhost:3000](http://localhost:3000)

   Use the username and password configured in the `docker-compose.yml` file to login and access the Langfuse project.

6. **:broom: Stop and Clean Up**

   To stop the running containers:

   ```bash
   sudo docker compose down
   ```

   To remove volumes and networks as well:

   ```bash
   sudo docker compose down -v
   ```
