using AuthNuget.Security;
using AuthService.Commands.Interfaces;
using AuthService.Commands.NewUser;
using AuthService.Commands.Seedwork;
using AuthService.Configurations;
using Microsoft.AspNetCore.Identity;
using Microsoft.Extensions.Options;

namespace AuthService.HostedServices;

public sealed class AddDefaultDbRecords : BackgroundService
{
    private readonly IOptions<DefaultAdmin> _defaultAdminConfig;
    private readonly IOptions<DefaultClient> _defaultClientSetting;
    private readonly IServiceProvider _serviceProvider;

    public AddDefaultDbRecords(
        IServiceProvider serviceProvider, 
        IOptions<DefaultAdmin> defaultAdminConfig, 
        IOptions<DefaultClient> defaultClientSetting)
    {
        _serviceProvider = serviceProvider;
        _defaultAdminConfig = defaultAdminConfig;
        _defaultClientSetting = defaultClientSetting;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        using var scope = _serviceProvider.CreateScope();

        scope.ServiceProvider.GetRequiredService<IMigrateUserContext>().Migrate();

        var roleManager = scope.ServiceProvider.GetRequiredService<RoleManager<IdentityRole>>();

        await AddRolesAsync(roleManager);

        var commandDispatcher = scope.ServiceProvider.GetRequiredService<ICommandDispatcher>();

        await AddAdminAsync(stoppingToken, commandDispatcher);

        await AddDefaultClientAsync(stoppingToken, commandDispatcher);

        ServiceReady.Instance.Ready<AddDefaultDbRecords>();
    }

    private async Task AddRolesAsync(RoleManager<IdentityRole> roleManager)
    {
        string[] roleNames = [RoleConstants.AdminRole, RoleConstants.Client];

        foreach (var roleName in roleNames)
        {
            var roleExist = await roleManager.RoleExistsAsync(roleName);

            if (!roleExist)
            {
                await roleManager.CreateAsync(new IdentityRole(roleName));
            }
        }
    }

    private async Task AddAdminAsync(CancellationToken stoppingToken, ICommandDispatcher commandDispatcher)
    {
        var defaultAdmin = _defaultAdminConfig.Value;

        var command = new CreateUser(defaultAdmin.Username, defaultAdmin.Password, RoleConstants.AdminRole);

        _ = await commandDispatcher.DispatchAsync(command, stoppingToken);
    }

    private async Task AddDefaultClientAsync(CancellationToken stoppingToken, ICommandDispatcher commandDispatcher)
    {
        var client = _defaultClientSetting.Value;

        var command = new CreateUser(client.Username, client.Password, RoleConstants.Client);

        _ = await commandDispatcher.DispatchAsync(command, stoppingToken);
    }
}