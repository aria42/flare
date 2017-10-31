(defproject tensors "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :global-vars {*warn-on-reflection* true}
  :jvm-opts ["-mx2000m" "-XX:+UnlockDiagnosticVMOptions" "-XX:+DebugNonSafepoints" "-XX:+UnlockCommercialFeatures" "-XX:+FlightRecorder" "-Djava.rmi.server.hostname=localhost" "-server"]
  :aot :all
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [prismatic/schema "1.1.6"]
                 [prismatic/plumbing "0.5.4"]
                 [uncomplicate/neanderthal "0.16.1"]
                 [org.apache.commons/commons-math3 "3.0"]
                 [org.clojure/tools.cli "0.3.5"]]
  :dev-dependencies [[criterium "0.4.4"]])
