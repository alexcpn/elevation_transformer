# Transformer based ITM Path Loss Engine

We will try to use Attention mechanism coupled with linear layers to go over a  sequencce of elevation data to get a learned attention score.
This along with ohter paramters we will feed to a multi layered deep neural network to predict the path loss and train on real data.

Thhe real data is generated from ITM code base for Wifi6 simulated Access Points

Irregular Terrain Model (ITM) (Longley-Rice) (20 MHz – 20 GHz) - https://its.ntia.gov/software/itm
 [source](https://github.com/Wireless-Innovation-Forum/Spectrum-Access-System/blob/master/src/harness/reference_models/propagation/itm/its/itm.cpp)

National Elevation Data- [Source](https://gdg.sc.egov.usda.gov/Catalog/ProductDescription/NED.html)

All Rights Reserved

