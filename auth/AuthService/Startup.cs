using System.Reflection;
using AuthNuget.Registration;
using AuthService.Commands.Interfaces;
using AuthService.Commands.NewUser;
using AuthService.Commands.Seedwork;
using AuthService.Configurations;
using AuthService.Controllers;
using AuthService.Dispatchers;
using AuthService.Domain;
using AuthService.HostedServices;
using AuthService.Middlewares;
using AuthService.Queries.Seedwork;
using AuthService.Repositories;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc.ApplicationParts;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using Microsoft.OpenApi.Models;
using RabbitMqNuget.Registration;

namespace AuthService;

public sealed class Startup
{
    private readonly IConfiguration _configuration;

    public Startup(IConfiguration configuration)
    {
        _configuration = configuration;
    }

    public void ConfigureServices(IServiceCollection services)
    {
        services.Configure<PasswordSettings>(_configuration.GetSection(nameof(PasswordSettings)));
        services.Configure<DefaultAdmin>(_configuration.GetSection($"Users:{nameof(DefaultAdmin)}"));
        services.Configure<DefaultClient>(_configuration.GetSection($"Users:{nameof(DefaultClient)}"));

        services.AddSingleton(serviceProvider => serviceProvider.GetRequiredService<IOptions<PasswordSettings>>().Value);

        RegisterConfiguration(services);
        RegisterInfrastructure(services);
        RegisterPresentation(services);
        RegisterApplication(services);

        services.AddEndpointsApiExplorer();

        services.AddSwaggerGen(c =>
        {
            c.AddSecurityDefinition("Bearer", new OpenApiSecurityScheme
            {
                Description = "JWT Authorization header using the Bearer scheme. Enter 'Bearer' [space] and then your token.",
                Name = "Authorization",
                In = ParameterLocation.Header,
                Type = SecuritySchemeType.Http,
                Scheme = "bearer",
                BearerFormat = "JWT"
            });

            c.AddSecurityRequirement(new OpenApiSecurityRequirement
            {
                {
                    new OpenApiSecurityScheme
                    {
                        Reference = new OpenApiReference
                        {
                            Type = ReferenceType.SecurityScheme,
                            Id = "Bearer",
                        }
                    },
                    new string[] {}
                }
            });
        });
    }

    public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
    {
        app.UseSwagger();
        app.UseSwaggerUI();

        app.UseHttpsRedirection();

        app.UseRouting();

        app.UseCors();

        app.UseAuthentication();
        app.UseAuthorization();

        app.UseMiddleware<ApiLoggingMiddleware>();

        app.UseEndpoints(endpoints =>
        {
            endpoints.MapControllers();
        });
    }

    private void RegisterApplication(IServiceCollection collection)
    {
        ScrutorScanForType(collection, typeof(IQueryHandler<,>), assemblyNames: "AuthService");
        ScrutorScanForType(collection, typeof(ICommandHandler<>), assemblyNames: "AuthService");
    }

    private void RegisterPresentation(IServiceCollection collection)
    {
        collection.AddHostedService<AddDefaultDbRecords>();

        collection.AddControllers()
            .AddJsonOptions(options =>
            {
                options.JsonSerializerOptions.PropertyNameCaseInsensitive = true;
            }).PartManager.ApplicationParts.Add(new AssemblyPart(typeof(AuthController).Assembly));

        collection
            .AddIdentity<UserPrincipal, IdentityRole>(options =>
            {
                _configuration.Bind("PasswordSettings", options.Password);

                options.SignIn.RequireConfirmedAccount = false;

                // Disable specific default validations
                options.Password.RequireDigit = false;
                options.Password.RequiredLength = 1;
                options.Password.RequireNonAlphanumeric = false;
                options.Password.RequireUppercase = false;
                options.Password.RequireLowercase = false;
                options.Password.RequiredUniqueChars = 0;
            })
            .AddEntityFrameworkStores<UserPrincipalContext>();

        collection.RegisterPfeAuthorization();
    }

    private void RegisterInfrastructure(IServiceCollection collection)
    {
        collection.AddScoped<IMigrateUserContext, UserPrincipalContext>(provider => provider.GetRequiredService<UserPrincipalContext>());

        collection.AddDbContext<UserPrincipalContext>(RepositoryDbContextOptionConfiguration);

        collection.RegisterMassTransit(
            _configuration.GetConnectionString("Rabbitmq") ?? throw new InvalidOperationException("Rabbitmq connection string is not found"),
            new MassTransitConfigurator().AddPublisher<UserCreated>("user-created-exchange"));
        
        return;

        void RepositoryDbContextOptionConfiguration(DbContextOptionsBuilder options)
        {
            var connectionString = _configuration.GetConnectionString("Postgres");

            options.EnableThreadSafetyChecks();
            options.UseNpgsql(connectionString);
        }
    }

    private void RegisterConfiguration(IServiceCollection collection)
    {
        collection.AddScoped<IQueryDispatcher, QueryDispatcher>();
        collection.AddScoped<ICommandDispatcher, CommandDispatcher>();
    }

    private void ScrutorScanForType(IServiceCollection services, Type type,
        ServiceLifetime lifetime = ServiceLifetime.Scoped, params string[] assemblyNames)
    {
        services.Scan(selector =>
        {
            selector.FromAssemblies(assemblyNames.Select(Assembly.Load))
                .AddClasses(filter => filter.AssignableTo(type))
                .AsImplementedInterfaces()
                .WithLifetime(lifetime);
        });
    }
}