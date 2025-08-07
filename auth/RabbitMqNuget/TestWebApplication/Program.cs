using RabbitMqNuget.Registration;

namespace TestWebApplication
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            var app = builder.Build();

            app.AddTransactionMiddleware();

            app.Run();
        }
    }
}
