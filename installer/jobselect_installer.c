// Installation Script for JobSelect CLI

#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <direct.h>

const char PYTHON_INSTALL[50] = "winget install python3";

const char REMAINING_COMMANDS[100] = "pip install jobselect && jobselect";

void install_py()
{
    printf("Installing Python...\n");

    int res_py = system(PYTHON_INSTALL);

    if (res_py == -1)
    {
        printf("Could not install python\n");
    }
}

int refresh_and_continue()
{
    char current_dir[MAX_PATH];

    // Get the current working directory so the new CMD opens in the exact same folder
    if (_getcwd(current_dir, sizeof(current_dir)) == NULL)
    {
        printf("Error getting current directory.\n");
        return 1;
    }

    printf("\nRefreshing environment variables to detect Python...\n");
    printf("Launching new CMD instance at: %s\n", current_dir);
    Sleep(1500);

    char parameters[256];
    snprintf(parameters, sizeof(parameters), "/k %s", REMAINING_COMMANDS);

    INT_PTR result = (INT_PTR)ShellExecute(
        NULL,        // No parent window
        "open",      // Operation to perform
        "cmd.exe",   // Target application
        parameters,  // Pass the commands to execute in the new window
        current_dir, // Working directory context
        SW_SHOW      // Show the window normally
    );

    if (result <= 32)
    {
        printf("Failed to start new CMD instance. Error code: %ld\n", (long)result);
        return 1;
    }

    ExitProcess(0);
}

int main()
{
    install_py();
    refresh_and_continue();

    return 0;
}
