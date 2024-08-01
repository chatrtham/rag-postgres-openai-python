# Deployment

## Deploying the App

To deploy the application using the Azure Developer CLI:

1. Navigate to the root directory of the project.
2. Run `azd deploy`.

## Updating Environment Variables for Azure Container App

To update environment variables for the Azure Container App, use the `az containerapp env update` command:

1. Open terminal or command prompt.
2. Run the following command, replacing placeholders with the actual values:

   ```bash
   az containerapp env update \
       --name <container-app-name> \
       --resource-group <resource-group-name> \
       --set-env-vars "<key1>=<value1>" "<key2>=<value2>" ...