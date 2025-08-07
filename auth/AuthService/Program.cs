using AuthNuget.Registration;
using AuthNuget.Security;

namespace AuthService;

public class Program
{
    private static IHostBuilder CreateHostBuilder(string[] args) => PfeSecureHost.Create<Startup>(args, RsaKeyStorage.Instance.PublicKey);

    public static void Main(string[] args)
    {
        CreateHostBuilder(args).Build().Run();
    }
}