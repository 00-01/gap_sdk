/*
 * Copyright (C) 2019 GreenWaves Technologies
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pmsis.h"

#include "bsp/bsp.h"
#include "bsp/gap9_v2.h"
#include "bsp/camera/himax.h"
#include "bsp/flash/hyperflash.h"
#include "bsp/ram/hyperram.h"
#include "bsp/ram/spiram.h"
#include "bsp/eeprom/24xx1025.h"
#include "bsp/eeprom/virtual_eeprom.h"

static int __bsp_init_pads_done = 0;

static void __bsp_init_pads()
{
  if (!__bsp_init_pads_done)
  {
    __bsp_init_pads_done = 1;
  }
}


int bsp_24xx1025_open(struct pi_24xx1025_conf *conf)
{
    pi_pad_set_function(CONFIG_24XX1025_I2C_SCL_PAD, CONFIG_24XX1025_I2C_SCL_PADFUN);
    pi_pad_set_function(CONFIG_24XX1025_I2C_SDA_PAD, CONFIG_24XX1025_I2C_SDA_PADFUN);

#ifdef CONFIG_24XX1025_I2C_SCL_PADMUX_GROUP
    pi_pad_set_mux_group(CONFIG_24XX1025_I2C_SCL_PAD, CONFIG_24XX1025_I2C_SCL_PADMUX_GROUP);
#endif

#ifdef CONFIG_24XX1025_I2C_SDA_PADMUX_GROUP
    pi_pad_set_mux_group(CONFIG_24XX1025_I2C_SDA_PAD, CONFIG_24XX1025_I2C_SDA_PADMUX_GROUP);
#endif

    return 0;
}

void bsp_24xx1025_conf_init(struct pi_24xx1025_conf *conf)
{
  conf->i2c_addr = CONFIG_24XX1025_I2C_ADDR;
  conf->i2c_itf = CONFIG_24XX1025_I2C_ITF;
}

void bsp_virtual_eeprom_conf_init(struct pi_virtual_eeprom_conf *conf)
{
  conf->i2c_addr = CONFIG_VIRTUAL_EEPROM_I2C_ADDR;
  conf->i2c_itf = CONFIG_VIRTUAL_EEPROM_I2C_ITF;
}

void bsp_hyperram_conf_init(struct pi_hyperram_conf *conf)
{
  conf->ram_start = CONFIG_HYPERRAM_START;
  conf->ram_size = CONFIG_HYPERRAM_SIZE;
  conf->skip_pads_config = 0;
  conf->hyper_itf = CONFIG_HYPERRAM_HYPER_ITF;
  conf->hyper_cs = CONFIG_HYPERRAM_HYPER_CS;
}


int bsp_hyperram_open(struct pi_hyperram_conf *conf)
{
  __bsp_init_pads();
  return 0;
}


void bsp_spiram_conf_init(struct pi_spiram_conf *conf)
{
  conf->ram_start = CONFIG_SPIRAM_START;
  conf->ram_size = CONFIG_SPIRAM_SIZE;
  conf->skip_pads_config = 0;
  conf->spi_itf = CONFIG_SPIRAM_SPI_ITF;
  conf->spi_cs = CONFIG_SPIRAM_SPI_CS;
}

int bsp_spiram_open(struct pi_spiram_conf *conf)
{
  return 0;
}


void bsp_aps25xxxn_conf_init(struct pi_aps25xxxn_conf *conf)
{
    conf->ram_start = CONFIG_APS25XXXN_START;
    conf->ram_size = CONFIG_APS25XXXN_SIZE;
    conf->spi_itf = CONFIG_APS25XXXN_SPI_ITF;
    conf->spi_cs = CONFIG_APS25XXXN_SPI_CS;
}

int bsp_aps25xxxn_open(struct pi_aps25xxxn_conf *conf)
{
    return 0;
}


void bsp_atxp032_conf_init(struct pi_atxp032_conf *conf)
{
    conf->spi_itf = CONFIG_ATXP032_SPI_ITF;
    conf->spi_cs = CONFIG_ATXP032_SPI_CS;
    conf->baudrate = 200000000;
}

int bsp_atxp032_open(struct pi_atxp032_conf *conf)
{
    return 0;
}


void bsp_spiflash_conf_init(struct pi_spiflash_conf *conf)
{
  conf->size = CONFIG_SPIFLASH_SIZE;
  // sector size is in number of KB
  conf->sector_size = CONFIG_SPIFLASH_SECTOR_SIZE;
  conf->spi_itf = CONFIG_SPIFLASH_SPI_ITF;
  conf->spi_cs = CONFIG_SPIFLASH_SPI_CS;
}

int bsp_spiflash_open(struct pi_spiflash_conf *conf)
{
  return 0;
}


void bsp_hyperflash_conf_init(struct pi_hyperflash_conf *conf)
{
  conf->hyper_itf = CONFIG_HYPERFLASH_HYPER_ITF;
  conf->hyper_cs = CONFIG_HYPERFLASH_HYPER_CS;
}

int bsp_hyperflash_open(struct pi_hyperflash_conf *conf)
{
  __bsp_init_pads();
  return 0;
}



void bsp_himax_conf_init(struct pi_himax_conf *conf)
{
  conf->i2c_itf = CONFIG_HIMAX_I2C_ITF;
  conf->cpi_itf = CONFIG_HIMAX_CPI_ITF;
}

int bsp_himax_open(struct pi_himax_conf *conf)
{
  __bsp_init_pads();
  return 0;
}

void bsp_nina_b112_conf_init(struct pi_nina_b112_conf *conf)
{
    conf->uart_itf = (uint8_t) CONFIG_NINA_B112_UART_ID;
}

int bsp_nina_b112_open(struct pi_nina_b112_conf *conf)
{
    return 0;
}

int bsp_nina_b112_open_old()
{
    __bsp_init_pads();
    return 0;
}

void bsp_init()
{
}


void pi_bsp_init_profile(int profile)
{
}



void pi_bsp_init()
{
  pi_bsp_init_profile(PI_BSP_PROFILE_DEFAULT);
}
